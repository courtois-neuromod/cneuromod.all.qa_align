"""End of run: do the events and the BOLD end together?

tail = BOLD length - end of the last event. A run is flagged when its events
go on after the BOLD has stopped, when its tail is unlike the other runs of
its task, or when the run is much shorter than the other runs of the same
stimulus.
"""

import numpy as np
import pandas as pd

from analysis import settings, signals

PAST = "events past the BOLD end"
CUT = "cut at the volume limit"
SHORT = "run shorter than the others of its stimulus"
UNUSUAL = "unusual tail for the task"
FLAGGED = (PAST, CUT, SHORT, UNUSUAL)
MIN_RUNS_AT_LIMIT = 2
SAME_TAIL = 2.0            # seconds; runs with the same unusual tail may be a task version


def fixed_length_identities(runs):
    """Stimuli whose runs normally all have the same length (not self-paced tasks)."""
    fixed = {}
    for identity, lengths in runs[runs.ident != ""].groupby("ident").n_vols:
        close = (lengths - lengths.median()).abs() <= 0.02 * lengths.median()
        fixed[identity] = len(lengths) >= 3 and float(close.mean()) >= 0.8
    return fixed


def volume_limit(runs):
    """The length at which scans were cut off, or None.

    Where runs have no fixed length, a scan that reaches the protocol's maximum
    is stopped there: several runs end at exactly the longest length, while
    most runs are shorter.
    """
    at_longest = runs.n_vols == runs.n_vols.max()
    if at_longest.sum() >= MIN_RUNS_AT_LIMIT and at_longest.mean() < 0.5:
        return runs.n_vols.max()
    return None


def unusual_tail(run, runs, spread, tail_min):
    """UNUSUAL, or 'ok (task version)' when several events files share the same tail."""
    threshold = max(tail_min, settings.TAIL_MAD * (spread or 0))
    if abs(run.tail_s - run.task_tail_median) <= threshold:
        return None
    peers = runs[(runs.task == run.task) & ((runs.tail_s - run.tail_s).abs() <= SAME_TAIL)]
    # a task version is several events files that agree; one shared (inherited) file is one file
    if len(peers) >= 3 and peers.events.nunique() > 1:
        return "ok (task version)"
    return UNUSUAL


def check(runs, tail_min):
    """Add events_end_s, tail_s and flag_end to the runs table.

    `runs` has columns id, task, ident, n_vols, tr, events, events_end_s and
    clock_confirmed_shift_s (seconds to add to the events when the clocks and
    the brain agree that the BOLD started early or late). `tail_min` is the
    smallest tail (seconds) that can be called unusual for this dataset.
    """
    runs = runs.copy()
    shifted_end = runs.events_end_s + runs.clock_confirmed_shift_s.fillna(0)
    runs["tail_s"] = (runs.n_vols * runs.tr - shifted_end).round(1)
    runs["task_tail_median"] = runs.groupby("task").tail_s.transform("median")
    spread = runs.groupby("task").tail_s.transform(lambda tails: signals.mad(tails.values))
    runs["ident_n_median"] = runs.ident.map(runs[runs.ident != ""].groupby("ident").n_vols.median())
    fixed, limit = fixed_length_identities(runs), volume_limit(runs)
    flags = []
    for position, run in enumerate(runs.itertuples()):
        has_tail = pd.notna(run.tail_s)
        if has_tail and run.tail_s < -run.tr and limit is not None and run.n_vols == limit:
            flags.append(CUT)
        elif has_tail and run.tail_s < -2 * run.tr:
            flags.append(PAST)
        elif fixed.get(run.ident, False) \
                and run.n_vols < settings.SHORT_RUN_FRACTION * run.ident_n_median:
            flags.append(SHORT)
        elif has_tail:
            flags.append(unusual_tail(run, runs, spread.iloc[position], tail_min) or "ok")
        else:
            flags.append("no events end")
    runs["flag_end"] = flags
    return runs


def summary(runs):
    """Per task: how the tails are spread (for the dashboard's reference table)."""
    rows = []
    for task, group in runs.dropna(subset=["tail_s"]).groupby("task"):
        tails = group.tail_s
        rows.append(dict(task=task, runs=len(group), median=round(float(tails.median()), 1),
                         p25=round(float(np.percentile(tails, 25)), 1),
                         p75=round(float(np.percentile(tails, 75)), 1)))
    return rows
