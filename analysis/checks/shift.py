"""Start shift: does the BOLD of a run start when its events say it does?

Each predictor gives the run a best shift. What counts is the shift relative
to the usual one of the task (the median over its runs, which absorbs the
haemodynamic delay and any constant offset of the task).

A predictor only votes for a task when it is RELIABLE there: most runs agree on
the same best shift. The run's result always comes from all reliable
predictors together, because a real shift moves (nearly) all of them:

  shift: clock + brain       the clocks show a shift and a predictor confirms it
  shift: most predictors     3+ predictors, all but one at most, same shift
  shift: both predictors     exactly 2 predictors, same shift
  shift: one predictor       a single predictor, with a clear gain in fit
  shift: clock only          the clocks show a shift the brain cannot confirm
  clocks and brain disagree  both see a shift, of different sizes
  no fit to its stimulus     every predictor far below its usual fit
  aligned (...)              nothing to report, with the reason when not plain
"""

import numpy as np
import pandas as pd

from analysis import settings

SHIFT_FLAGS = ("shift: clock + brain", "shift: most predictors", "shift: both predictors",
               "shift: one predictor", "shift: clock only", "clocks and brain disagree",
               "no fit to its stimulus")
ALIAS_PERIODS = (-3, -2, -1, 1, 2, 3)


def is_flagged(flag):
    return isinstance(flag, str) and flag in SHIFT_FLAGS


def reliability(fits):
    """One row per (task, predictor): usual shift, agreement between runs, reliable or not."""
    rows = []
    for (task, predictor), group in fits.groupby(["task", "pred"]):
        baseline = float(group.best_lag_s.median())
        within = (group.best_lag_s - baseline).abs() <= settings.SHIFT_MIN * group.tr
        concentration = float(within.mean())
        rows.append(dict(
            task=task, pred=predictor, runs=len(group), baseline_s=baseline,
            conc=round(concentration, 2), r_median=round(float(group.r_best.median()), 3),
            reliable=bool(concentration >= settings.CONCENTRATION_MIN
                          and len(group) >= settings.RELIABLE_MIN_RUNS)))
    table = pd.DataFrame(rows, columns=["task", "pred", "runs", "baseline_s", "conc", "r_median",
                                        "reliable"])
    return table.astype({"reliable": bool})    # stays a boolean mask when there are no rows


def secondary_peaks(group, baseline, lags, tr):
    """Shifts (volumes) at which the typical curve of a periodic design peaks again."""
    typical = np.median(np.array([np.asarray(curve, float) for curve in group.curve]), axis=0)
    floor = np.median(typical)
    main = typical[int(np.argmin(abs(lags - baseline)))] - floor
    peaks = []
    for position in range(1, len(typical) - 1):
        is_peak = typical[position] >= typical[position - 1] \
            and typical[position] >= typical[position + 1]
        far = abs(lags[position] - baseline) > settings.SHIFT_MIN * tr
        if is_peak and far and main > 0 and (typical[position] - floor) >= 0.5 * main:
            peaks.append(round(float((lags[position] - baseline) / tr), 1))
    return peaks


def on_alias(shift, period, peaks):
    """True when a shift lands on a whole number of trial periods or a secondary peak."""
    if pd.notna(period):
        for count in ALIAS_PERIODS:
            if abs(shift - count * period) <= max(0.75, 0.15 * abs(count * period)):
                return True
    return any(abs(shift - peak) <= max(settings.SHIFT_MIN, 0.1 * abs(peak)) for peak in peaks)


def annotate(fits, reliable_table, lags, trial_periods):
    """Add to each (run, predictor) fit its shift, gain and alias status.

    `fits` has columns id, pred, task, tr, best_lag_s, r_best, curve.
    """
    if fits.empty:                       # no predictor at all (e.g. empty events files)
        columns = ["baseline_s", "reliable", "r_usual", "shift_vols", "r_base", "gain",
                   "period_vols", "aliased"]
        empty = fits.assign(**{name: [] for name in columns}).astype({"reliable": bool})
        return empty, reliable_table.assign(alias_vols=[])
    by_key = reliable_table.set_index(["task", "pred"])
    keys = list(zip(fits.task, fits.pred))
    fits = fits.copy()
    fits["baseline_s"] = [by_key.baseline_s[key] for key in keys]
    fits["reliable"] = [bool(by_key.reliable[key]) for key in keys]
    fits["r_usual"] = [by_key.r_median[key] for key in keys]
    fits["shift_vols"] = ((fits.best_lag_s - fits.baseline_s) / fits.tr).round(1)
    fits["r_base"] = [float(np.asarray(curve)[np.argmin(abs(lags - baseline))])
                      for curve, baseline in zip(fits.curve, fits.baseline_s)]
    fits["gain"] = (fits.r_best - fits.r_base).round(3)
    peaks = {key: secondary_peaks(group, by_key.baseline_s[key], lags, float(group.tr.iloc[0]))
             for key, group in fits.groupby(["task", "pred"])}
    fits["period_vols"] = fits.id.map(trial_periods) / fits.tr
    fits["aliased"] = [on_alias(shift, period, peaks[key])
                       for shift, period, key in zip(fits.shift_vols, fits.period_vols, keys)]
    reliable_table = reliable_table.copy()
    reliable_table["alias_vols"] = [peaks.get(key, []) for key in
                                    zip(reliable_table.task, reliable_table.pred)]
    return fits, reliable_table


def simulated_gain(usable, lags, simulated):
    """Gain each predictor would show if the run were shifted by `simulated` volumes."""
    gains = []
    for curve, baseline, tr in zip(usable.curve, usable.baseline_s, usable.tr):
        curve = np.asarray(curve)
        moved = curve[np.argmin(abs(lags - (baseline - simulated * tr)))]
        gains.append(float(curve.max() - moved))
    return pd.Series(gains, index=usable.index)


def brain_verdict(usable, shifts, gains):
    """(flag, shift, predictors that agree) from the predictors alone."""
    middle = float(np.median(shifts))
    agree = [name for name, shift in zip(usable.pred, shifts)
             if abs(shift - middle) <= settings.shift_tolerance(middle)]
    count = len(usable)
    if abs(middle) >= settings.SHIFT_FLAG:
        if count >= 3 and len(agree) >= count - 1:
            return "shift: most predictors", round(middle, 1), agree
        if count == 2 and len(agree) == 2:
            return "shift: both predictors", round(middle, 1), agree
        if count == 1 and float(gains.iloc[0]) >= settings.GAIN_ONE:
            return "shift: one predictor", round(middle, 1), agree
    if any(abs(shift) >= settings.SHIFT_MIN for shift in shifts):
        return "aligned (shift in few predictors)", round(middle, 1), []
    return "aligned", round(middle, 1), []


def with_clock(usable, clock_shift, brain):
    """Combine the clocks' shift with the predictors' verdict."""
    confirming = usable[((usable.shift_vols - clock_shift).abs()
                         <= settings.shift_tolerance(clock_shift))
                        & (usable.gain >= settings.GAIN_CONFIRM)]
    if len(confirming):
        return ("shift: clock + brain", round(float(confirming.shift_vols.median()), 1),
                list(confirming.pred))
    if brain[0].startswith("shift"):
        return "clocks and brain disagree", brain[1], brain[2]
    return "shift: clock only", None, []


def combine(run_fits, lags, clock_shift=np.nan, simulated=0.0):
    """The run's start-shift result from its reliable predictors (and the clocks).

    Returns (flag, shift in volumes or None, comma-separated agreeing predictors).
    `simulated` adds a shift to every predictor, to measure what the rule catches.
    """
    has_clock = pd.notna(clock_shift) and abs(clock_shift) >= settings.SHIFT_MIN and not simulated
    if not simulated and len(run_fits) and \
            (run_fits.r_best < settings.NO_FIT_FRACTION * run_fits.r_usual).all() and not has_clock:
        return "no fit to its stimulus", None, ""
    usable = run_fits[run_fits.r_best >= settings.R_MIN]
    if usable.empty:
        return ("shift: clock only" if has_clock else "aligned (weak brain signal)"), None, ""
    if not simulated and not has_clock and usable.aliased.mean() >= 0.5:
        return "aligned (periodic design)", round(float(usable.shift_vols.median()), 1), ""
    shifts = usable.shift_vols + simulated
    gains = simulated_gain(usable, lags, simulated) if simulated else usable.gain
    brain = brain_verdict(usable, shifts, gains)
    flag, shift, agree = with_clock(usable, clock_shift, brain) if has_clock else brain
    return flag, shift, ",".join(agree)


def responsible_file(flag, backers, run_fits):
    """Which file a shift points at, using the other subjects as a witness."""
    if not flag.startswith("shift") or not backers:
        return ""
    names = backers.split(",")
    from_events = [name for name in names if name != "isc"]
    others = run_fits[run_fits.pred == "isc"]
    others_aligned = len(others) and float(others.r_best.iloc[0]) >= settings.R_MIN \
        and abs(float(others.shift_vols.iloc[0])) < settings.SHIFT_MIN
    if others_aligned and from_events:
        return "events file (other subjects aligned)"
    if "isc" in names and from_events:
        return "BOLD run (events and other subjects agree)"
    if names == ["isc"]:
        return "BOLD run (other subjects)"
    return ""


def blind_to_simulation(run_fits):
    """A shift of the simulated size is invisible when it is a whole number of trials."""
    periods = run_fits.period_vols.dropna()
    if not len(periods):
        return False
    period = periods.iloc[0]
    size = float(settings.SIMULATED_VOLUMES)
    return any(abs(size - count * period) <= max(0.75, 0.15 * count * period)
               for count in (1, 2, 3))


def check(fits, run_ids, lags, clock_shifts):
    """Start-shift result of every run.

    Returns (rows, sensitivity): one dict per run id, and the share of aligned
    runs in which a simulated shift of settings.SIMULATED_VOLUMES is caught.
    """
    reliable = fits[fits.reliable]
    by_run = {run_id: group for run_id, group in reliable.groupby("id")}
    all_by_run = {run_id: group for run_id, group in fits.groupby("id")}
    empty = reliable.iloc[:0]
    rows, caught = [], []
    for run_id in run_ids:
        run_fits = by_run.get(run_id, empty)
        clock_shift = clock_shifts.get(run_id, np.nan)
        flag, shift, backers = combine(run_fits, lags, clock_shift)
        row = dict(id=run_id, flag_shift=flag, shift_vols=shift, backers=backers,
                   n_reliable=len(run_fits), clock_shift_vols=clock_shift,
                   where=responsible_file(flag, backers, run_fits))
        for fit in all_by_run.get(run_id, empty).itertuples():
            row[f"shift_{fit.pred}"] = fit.shift_vols
            row[f"r_{fit.pred}"] = round(fit.r_best, 3)
            row[f"gain_{fit.pred}"] = fit.gain
        rows.append(row)
        if flag == "aligned" and len(run_fits):
            hit = not combine(run_fits, lags, simulated=settings.SIMULATED_VOLUMES)[0] \
                .startswith("aligned")
            caught.append(False if blind_to_simulation(run_fits) else hit)
    sensitivity = round(100 * float(np.mean(caught)), 1) if caught else None
    return rows, sensitivity
