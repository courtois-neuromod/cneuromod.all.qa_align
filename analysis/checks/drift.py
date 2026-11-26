"""Drift and jumps: does the shift stay the same from the start of a run to its end?

The scan gives the best shift in sliding windows along the run. Here each
predictor's slope (volumes per 10 minutes, weighted by the fit of each window)
is standardised per task and predictor, and averaged over the reliable
predictors: the run's drift score. A run is flagged when the score is large,
the predictors (all but one at most) slope the same way, and the mean slope is
at least settings.SLOPE_MIN.

The same measure on runs with a simulated jump or drift tells how much of each
the check would catch; on the real runs it gives the rate of flags.
"""

import numpy as np
import pandas as pd

from analysis import settings, signals

FLAGGED = "drift or jump"
FINE = "steady"
MIN_WINDOWS = 8
SPLIT_MARGIN = 4           # windows around a split point share data with it and are left out


def slope(windows, tr):
    """Slope of the windowed shift, in volumes per 10 minutes (fit-weighted)."""
    windows = np.array(windows)
    if len(windows) < MIN_WINDOWS:
        return np.nan
    time, shift, fit = windows[:, 0], windows[:, 1], np.clip(windows[:, 2], 0, None)
    weight = np.sqrt(fit ** 2 + 1e-6)
    design = np.c_[np.ones_like(time), time / 600]
    solution = np.linalg.lstsq(design * weight[:, None], shift * weight, rcond=None)[0]
    return float(solution[1]) / tr


def largest_step(windows, tr):
    """(size in volumes, time in seconds) of the largest change of the shift, or (nan, nan)."""
    windows = np.array(windows)
    if len(windows) < MIN_WINDOWS:
        return np.nan, np.nan
    time, shift = windows[:, 0], windows[:, 1]
    best, when = 0.0, np.nan
    for split in range(SPLIT_MARGIN, len(shift) - SPLIT_MARGIN + 1):
        before = shift[:max(split - SPLIT_MARGIN, 1)]
        after = shift[split + SPLIT_MARGIN + 1:] if split + SPLIT_MARGIN + 1 < len(shift) \
            else shift[-1:]
        change = np.median(after) - np.median(before)
        if abs(change) > abs(best):
            best, when = change, time[split]
    return round(float(best) / tr, 2), float(when)


def score(group, n_reliable):
    """(drift score, flagged) of one run from its reliable predictors' rows."""
    group = group.dropna(subset=["z"])
    if len(group) == 0 or len(group) < min(2, n_reliable):
        return None, False
    mean = float(group.z.mean())
    same_way = int((np.sign(group.z) == np.sign(mean)).sum())
    flagged = abs(mean) >= settings.DRIFT_Z \
        and same_way >= len(group) - (len(group) >= 3) \
        and abs(float(group.slope.mean())) >= settings.SLOPE_MIN
    return mean, flagged


def measures(drift, task_of, reliable_keys, tr):
    """Slope, largest step and standardised slope of every reliable (run, predictor, variant)."""
    table = pd.DataFrame(drift)
    table["slope"] = [slope(windows, tr) for windows in table.windows]
    table["task"] = table.id.map(task_of)
    keep = [(task, predictor) in reliable_keys for task, predictor in zip(table.task, table.pred)]
    table = table[np.array(keep) & table.slope.notna()].copy()
    table["z"] = np.nan
    for (task, predictor), group in table[table.variant == "real"].groupby(["task", "pred"]):
        centre, spread = group.slope.median(), signals.mad(group.slope.values) or 1e-3
        same = (table.task == task) & (table.pred == predictor)
        table.loc[same, "z"] = (table.loc[same, "slope"] - centre) / spread
    return table


def rate(table, variant, n_reliable):
    """Share of runs of one variant (real, jump, drift) that the rule flags, in percent."""
    hits = []
    for _, group in table[table.variant == variant].groupby("id"):
        value, flagged = score(group, n_reliable[group.task.iloc[0]])
        if value is not None:
            hits.append(flagged)
    return round(100 * float(np.mean(hits)), 1) if hits else None


def check(drift, task_of, reliable_keys, tr):
    """Drift result per run id, the measures table, and the simulation rates."""
    rates = dict(drift_jump=None, drift_drift=None, drift_false_alarm=None)
    if not len(drift):
        return {}, pd.DataFrame(), rates
    table = measures(drift, task_of, reliable_keys, tr)
    n_reliable = {task: sum(1 for key in reliable_keys if key[0] == task)
                  for task in table.task.unique()}
    results = {}
    for run_id, group in table[table.variant == "real"].groupby("id"):
        value, flagged = score(group, n_reliable[group.task.iloc[0]])
        if value is None:
            continue
        results[run_id] = dict(drift_score=round(value, 2),
                               flag_drift=FLAGGED if flagged else FINE)
        for row in group.itertuples():
            results[run_id][f"slope_{row.pred}"] = round(row.slope, 2)
    rates = dict(drift_jump=rate(table, "jump", n_reliable),
                 drift_drift=rate(table, "drift", n_reliable),
                 drift_false_alarm=rate(table, "real", n_reliable))
    return results, table, rates
