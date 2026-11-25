"""Repetitions (videogame datasets): is each play's replay the right one?

Every play (a game row of the events) points at a replay file. The candidates
for a play are the replays of every play of the same level in the same session.
Two pieces of evidence:

slot   the replay must fill the play's slot in the events:
       next play's onset - onset = replay length + the session's usual gap.
brain  inside the run, only this play's part of each predictor is replaced by
       the candidate's, and the fit with the BOLD (the run's top parcels, at
       its best shift) is compared with the fit of the play's own replay.

This module measures; `analysis.checks.repetition_flags` decides.
"""

import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import datasets, loading, settings, signals
from analysis import scan as scan_module
from analysis.predictors import games

BRAIN_PREDICTORS = ["resp", "luminance", "flow", "audio"]   # game on/off is the slot test
STATE = {}


def session_plays(runs, info, bids_root, cache):
    """Every play of one session, with its slot and its replay's content."""
    plays = []
    for run in runs.itertuples():
        listed = info.get(run.id, {}).get("plays", [])
        # one frame rate per run (the median over its plays): a play whose replay has the
        # wrong length must not be allowed to explain it with a frame rate of its own
        frame_rate = float(np.median([play[3] for play in listed])) if listed else np.nan
        for position, (onset, _, _, _, stim_file, level) in enumerate(listed):
            following = listed[position + 1][0] if position + 1 < len(listed) else np.nan
            if stim_file not in cache:
                cache[stim_file] = games.load_recording(bids_root, stim_file)
            plays.append(dict(run_id=run.id, run=run.run, position=position, onset=onset,
                              level=level, stim=stim_file, frame_rate=frame_rate,
                              slot=following - onset, replay=cache[stim_file]))
    return plays


def replay_length(play, replay=None):
    replay = replay or play["replay"]
    return replay["n_frames"] / play["frame_rate"]


def usual_gap(plays):
    """Median time between the end of a replay and the next play, in this session."""
    gaps = [play["slot"] - replay_length(play) for play in plays
            if play["replay"] and pd.notna(play["slot"])]
    return float(np.median(gaps)) if gaps else np.nan


def play_signal(predictor, replay, frame_rate, length, feature_means):
    """Unconvolved signal of one play on the predictor grid, cut at `length` seconds."""
    if predictor not in replay:
        return None
    n_samples = int(round(length / settings.GRID_STEP))
    values = replay[predictor]
    index = (np.arange(len(values)) / frame_rate / settings.GRID_STEP).astype(int)
    keep = index < n_samples
    signal = np.zeros(n_samples)
    if predictor == "resp":
        return signal + np.bincount(index[keep], values[keep], n_samples) / settings.GRID_STEP / 10
    total = np.bincount(index[keep], values[keep], n_samples)
    count = np.bincount(index[keep], None, n_samples)
    signal[count > 0] = total[count > 0] / count[count > 0] - feature_means.get(predictor, 0.0)
    return signal


def fit_with_replay(play, candidate, predictor, length, feature_means):
    """Fit of the run's BOLD when this play's part of the predictor comes from `candidate`."""
    key = (play["run_id"], predictor)
    if key not in STATE["parcels"] or key not in STATE["best_lag"]:
        return None
    signal = STATE["predictors"][play["run_id"]][predictor].astype(float)
    own = play["replay"]
    if candidate is not own:
        own_part = play_signal(predictor, own, play["frame_rate"], length, feature_means)
        other_part = play_signal(predictor, candidate, play["frame_rate"], length, feature_means)
        if own_part is None or other_part is None:
            return None
        change = np.convolve(other_part - own_part, signals.HRF)
        start = signals.grid_index(play["onset"])
        stop = min(len(signal), start + len(change))
        signal = signal.copy()
        signal[start:stop] += change[:stop - start]
    data = STATE["bold"][play["run_id"]]
    n_volumes = len(data) + settings.DROPPED_VOLUMES
    regressor = signals.lagged_regressors(signal, n_volumes, STATE["tr"],
                                          [STATE["best_lag"][key]])[0]
    return float((data[:, STATE["parcels"][key]].T @ regressor).mean() / n_volumes)


def measure_play(play, plays, gap):
    """One row: the play's slot fit, and each candidate replay's slot and brain fit."""
    own = play["replay"]
    known_slot = pd.notna(play["slot"]) and pd.notna(gap)
    row = dict(run_id=play["run_id"], run=play["run"], play=play["position"] + 1,
               level=play["level"], stim=Path(play["stim"]).name, cand=[],
               slot_dev=round(play["slot"] - replay_length(play) - gap, 2)
               if own and known_slot else None)
    if own is None or play["run_id"] not in STATE["bold"]:
        return row
    length = play["slot"] if pd.notna(play["slot"]) else \
        replay_length(play) + (gap if pd.notna(gap) else 0)
    feature_means = {name: float(np.mean(own[name])) for name in games.FEATURES if name in own}
    own_fit = {name: fit_with_replay(play, own, name, length, feature_means)
               for name in BRAIN_PREDICTORS}
    for other in plays:
        replay = other["replay"]
        if other is play or other["level"] != play["level"] or replay is None \
                or other["stim"] == play["stim"]:
            continue
        fits = {name: fit_with_replay(play, replay, name, length, feature_means)
                for name in BRAIN_PREDICTORS}
        row["cand"].append(dict(
            stim=Path(other["stim"]).name,
            slot_dev=round(play["slot"] - replay_length(play, replay) - gap, 2)
            if known_slot else None,
            d={name: None if fits[name] is None or own_fit[name] is None
               else round(fits[name] - own_fit[name], 4) for name in BRAIN_PREDICTORS}))
    return row


def subject_parcels(runs):
    """{(run id, predictor): top parcels}, picked leave-one-run-out as in the scan."""
    parcels = {}
    for predictor in BRAIN_PREDICTORS:
        found = {run_id: scan_module.EventSignal(
                     STATE["predictors"][run_id][predictor],
                     len(STATE["bold"][run_id]) + settings.DROPPED_VOLUMES, STATE["tr"])
                 for run_id in runs.id if predictor in STATE["predictors"].get(run_id, {})}
        if len(found) < settings.MIN_RUNS_PER_PREDICTOR:
            continue
        zero_fits = {run_id: signal.fit_at_zero(STATE["bold"][run_id])
                     for run_id, signal in found.items()}
        for run_id in found:
            parcels[(run_id, predictor)] = scan_module.top_parcels(zero_fits, run_id)
    return parcels


def check_subject(subject):
    index = STATE["index"]
    runs = index[index["sub"] == subject]
    STATE["parcels"] = subject_parcels(runs)
    cache, rows = {}, []
    for session, session_runs in runs.groupby("ses"):
        plays = session_plays(session_runs, STATE["info"], STATE["bids_root"], cache)
        gap = usual_gap(plays)
        for play in plays:
            row = measure_play(play, plays, gap)
            row.update(sub=subject, ses=session)
            rows.append(row)
    return rows


def check_dataset(checkout, dataset, scan_results, subjects=None, n_jobs=6):
    """Measure every play of a videogame dataset. Returns dict(dataset, plays)."""
    listed = pd.DataFrame(scan_results["index"])
    index, bold, predictors, info, tr, _ = loading.load_dataset(checkout, dataset, listed,
                                                                 subjects)
    best_lag = {(fit["id"], fit["pred"]): fit["best_lag_s"] for fit in scan_results["runs"]}
    STATE.update(index=index, bold=bold, predictors=predictors, info=info, tr=tr,
                 best_lag=best_lag, bids_root=datasets.bids_dir(checkout, dataset))
    subject_list = sorted(index["sub"].unique())
    plays = []
    if subject_list:
        with Pool(max(1, min(n_jobs, len(subject_list)))) as pool:
            for rows in pool.imap_unordered(check_subject, subject_list):
                plays += rows
    STATE.clear()
    return dict(dataset=dataset, plays=plays)


def write(results, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)


def read(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
