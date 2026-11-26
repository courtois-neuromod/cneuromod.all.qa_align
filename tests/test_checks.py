"""The rules of each check, on small hand-made inputs."""

import numpy as np
import pandas as pd

from analysis import bids_index, known_issues, settings, signals
from analysis.checks import clock, drift, endrun, pairing, repetition_flags, shift
from analysis.predictors import events as event_predictors

LAGS = settings.LAGS[::settings.CURVE_STEP]
TR = settings.DEFAULT_TR


def fit_rows(shifts, gain=0.1, r_best=0.3, aliased=False):
    """Rows of one run's reliable predictors, each peaking at its shift."""
    rows = []
    for name, value in shifts.items():
        curve = np.full(len(LAGS), r_best - gain)
        curve[np.argmin(abs(LAGS - value * TR))] = r_best
        rows.append(dict(pred=name, shift_vols=value, gain=gain, r_best=r_best, r_usual=r_best,
                         aliased=aliased, curve=curve, baseline_s=0.0, tr=TR,
                         period_vols=np.nan))
    return pd.DataFrame(rows)


# --- signals ---------------------------------------------------------------
def test_hrf_sums_to_one_and_peaks_after_a_few_seconds():
    assert abs(signals.HRF.sum() - 1) < 1e-9
    assert 3 < np.argmax(signals.HRF) * settings.GRID_STEP < 8


def test_lagged_regressor_moves_later_with_a_positive_lag():
    n_samples = signals.grid_length(200, TR)
    predictor = signals.convolve_hrf(signals.boxcar([50.0], [10.0], n_samples))
    now, later = signals.lagged_regressors(predictor, 200, TR, [0.0, 15.0])
    assert np.argmax(later) - np.argmax(now) == round(15.0 / TR)


# --- predictors from events -------------------------------------------------
def test_events_predictors_split_stimuli_conditions_and_responses():
    table = pd.DataFrame({
        "onset": [0, 10, 20, 30, 40, 50, 12, 32, 52],
        "duration": [5, 5, 5, 5, 5, 5, 0, 0, 0],
        "trial_type": ["face", "house", "face", "house", "face", "house",
                       "response", "response", "response"]})
    predictors, info = event_predictors.build(table, 100, TR)
    assert {"stim", "cond1", "cond2", "resp"} <= set(predictors)
    assert info["events_end"] == 55
    assert set(info["labels"].values()) == {"trial_type=face", "trial_type=house"}


def test_naturalistic_run_lasts_until_the_end_of_the_bold():
    predictors, info = event_predictors.build(pd.DataFrame({"onset": [0.0]}), 100, TR)
    assert "stim" in predictors
    assert info["events_end"] == 100 * TR


# --- start shift ------------------------------------------------------------
def test_aligned_run_is_not_flagged():
    assert shift.combine(fit_rows({"stim": 0.2, "resp": -0.3, "isc": 0.0}), LAGS)[0] == "aligned"


def test_converging_predictors_flag_a_shift():
    flag, size, backers = shift.combine(fit_rows({"stim": 4.0, "resp": 4.4, "isc": 3.8}), LAGS)
    assert flag == "shift: most predictors"
    assert size == 4.0
    assert set(backers.split(",")) == {"stim", "resp", "isc"}


def test_one_predictor_out_of_three_does_not_flag():
    flag, _, _ = shift.combine(fit_rows({"stim": 5.0, "resp": 0.1, "isc": 0.0}), LAGS)
    assert flag == "aligned (shift in few predictors)"


def test_a_lone_predictor_needs_a_clear_gain():
    assert shift.combine(fit_rows({"stim": 5.0}, gain=0.1), LAGS)[0] == "shift: one predictor"
    assert not shift.is_flagged(shift.combine(fit_rows({"stim": 5.0}, gain=0.01), LAGS)[0])


def test_clock_shift_is_confirmed_by_a_predictor_or_reported_alone():
    confirmed = shift.combine(fit_rows({"stim": -5.1, "resp": 0.0}), LAGS, clock_shift=-5.0)
    assert confirmed[0] == "shift: clock + brain"
    alone = shift.combine(fit_rows({"stim": 0.1, "resp": 0.0}), LAGS, clock_shift=-5.0)
    assert alone[0] == "shift: clock only"


def test_weak_signal_and_periodic_designs_are_not_shifts():
    weak = shift.combine(fit_rows({"stim": 6.0}, r_best=0.01, gain=0.0), LAGS)
    assert weak[0] == "no fit to its stimulus" or weak[0] == "aligned (weak brain signal)"
    periodic = shift.combine(fit_rows({"stim": 3.0, "resp": 3.0}, aliased=True), LAGS)
    assert periodic[0] == "aligned (periodic design)"


def test_predictor_is_reliable_only_when_runs_agree():
    agree = pd.DataFrame(dict(task="t", pred="stim", tr=TR, best_lag_s=[5.0] * 9 + [20.0],
                              r_best=0.3))
    scattered = agree.assign(pred="resp", best_lag_s=np.linspace(-20, 40, 10))
    table = shift.reliability(pd.concat([agree, scattered])).set_index("pred")
    assert bool(table.reliable["stim"]) and not bool(table.reliable["resp"])


# --- clock ------------------------------------------------------------------
def test_task_clock_column_is_detected():
    table = pd.DataFrame({"onset": [1.0, 5.0, 9.0], "sample": [1001.0, 1005.0, 1009.0],
                          "duration": [2.0, 2.0, 2.0]})
    assert clock.events_clock_origin(table) == 1000.0
    assert clock.events_clock_origin(table.drop(columns="sample")) is None


def test_clock_shift_is_the_deviation_from_the_session_in_volumes():
    runs = pd.DataFrame({
        "sub": "sub-01", "ses": "ses-001",
        "acq_time": ["10:00:00", "10:10:00", "10:20:00", "10:30:00"],
        "clock_origin": [100.0, 700.0, 1300.0, 1900.0 + 5 * TR]})
    shifts = clock.clock_shifts(runs, TR)
    assert list(shifts[:3]) == [0.0, 0.0, 0.0]
    assert shifts[3] == 5.0     # the task started 5 volumes after the BOLD


def test_a_column_that_is_not_a_steady_clock_gives_no_evidence():
    table = pd.DataFrame({"onset": [1.0, 5.0, 9.0], "offset_flip": [4.0, 8.0, 12.0]})
    assert clock.events_clock_origin(table) is None
    runs = pd.DataFrame({
        "sub": "sub-01", "ses": "ses-001",
        "acq_time": ["10:00:00", "10:10:00", "10:20:00", "10:30:00"],
        "clock_origin": [3.0, 3.0, 3.0, 3.0]})      # same value in every run: not a clock
    assert clock.clock_shifts(runs, TR).isna().all()


# --- pairing ----------------------------------------------------------------
def test_pairing_flags_when_the_primary_predictors_prefer_one_other_run():
    group = pd.DataFrame(dict(pred=["stim", "resp"], margin=[-0.2, -0.1], z=[-4.0, -3.0],
                              other=["run B", "run B"]))
    assert pairing.run_result(group)["flag_pair"] == pairing.FLAGGED
    own = group.assign(margin=[0.1, -0.1])
    assert pairing.run_result(own)["flag_pair"] == pairing.FINE


# --- drift ------------------------------------------------------------------
def test_slope_of_a_drifting_run_is_measured_in_volumes_per_ten_minutes():
    time = np.arange(20) * 30.0
    steady = [[t, 0.0, 0.3] for t in time]
    drifting = [[t, 2 * TR * t / 600, 0.3] for t in time]
    assert abs(drift.slope(steady, TR)) < 1e-9
    assert abs(drift.slope(drifting, TR) - 2.0) < 1e-6


def test_largest_step_finds_a_jump():
    windows = [[i * 15.0, 0.0 if i < 12 else 3 * TR, 0.3] for i in range(24)]
    size, _ = drift.largest_step(windows, TR)
    assert size == 3.0


# --- end of run -------------------------------------------------------------
def test_end_of_run_flags():
    runs = pd.DataFrame(dict(
        id=list("abcdef"), task="t", ident="", n_vols=[300, 300, 300, 300, 300, 300], tr=TR,
        events=[f"{name}.tsv" for name in "abcdef"],
        events_end_s=[440.0, 441.0, 439.0, 440.0, 460.0, 300.0],
        clock_confirmed_shift_s=np.nan))
    flagged = endrun.check(runs, tail_min=15.0).set_index("id").flag_end
    assert flagged["a"] == "ok"
    assert flagged["e"] == endrun.PAST
    assert flagged["f"] == endrun.UNUSUAL


# --- repetitions --------------------------------------------------------------
def test_replay_swap_is_recognised_from_the_slots():
    def play(name, deviation, other, other_deviation):
        return dict(id=name, sub="s", ses="1", stim=name, flag=repetition_flags.BAD_SLOT,
                    slot_dev=deviation,
                    cand=[dict(stim=other, slot_dev=other_deviation, z=None, hit=False)])
    rows = [play("a", 80.0, "b", 0.3), play("b", -80.0, "a", -0.2)]
    assert repetition_flags.find_swaps(rows) == {"a": "b", "b": "a"}


# --- index and known issues ---------------------------------------------------
def test_events_are_inherited_from_the_closest_matching_file(tmp_path):
    (tmp_path / "task-movie_events.tsv").write_text("onset\tduration\n0\t10\n")
    by_folder = {"": ["task-movie_events.tsv"]}
    bold = "sub-01/ses-001/func/sub-01_ses-001_task-movie_bold.nii.gz"
    found = bids_index.closest_events(bold, bids_index.entities(bold), by_folder, tmp_path)
    assert found == "task-movie_events.tsv"
    other = bold.replace("task-movie", "task-other")
    assert bids_index.closest_events(other, bids_index.entities(other), by_folder, tmp_path) == ""


def test_session_and_run_numbers_ignore_zero_padding():
    assert bids_index.run_key(bids_index.entities("sub-01_ses-002_task-x_run-03_bold")) == \
        bids_index.run_key(bids_index.entities("sub-01_ses-02_task-x_run-3_bold"))


def test_known_issues_table_is_well_formed_and_matches_runs():
    table = known_issues.load()
    assert (table.url.str.startswith("https://")).all()
    assert not table[["dataset", "sub", "ses", "runs", "reference", "note"]].eq("").any().any()
    runs = pd.DataFrame(dict(sub=["sub-01", "sub-01"], ses=["ses-004", "ses-004"], run=["2", "3"]))
    annotated = known_issues.annotate(runs, "mario")
    assert "mario#24" in annotated.known[0] and annotated.known[1] == ""
