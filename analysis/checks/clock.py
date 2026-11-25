"""Clock evidence for a shifted BOLD start, with no brain signal involved.

Some events files keep the task's own clock next to `onset` (a column whose
difference with `onset` is the same on every row: the task-clock time of the
run's t=0). The scanner writes the time of day of the first volume in the BOLD
sidecar (AcquisitionTime). Within a session both clocks run steadily, so
`AcquisitionTime - task-clock origin` is the same for every run of the session
unless one run's BOLD did not start with the task.
"""

import numpy as np
import pandas as pd

from analysis import settings

PREFERRED_COLUMNS = ["sample", "onset_abs", "onset_time", "time", "flip_time"]
SKIPPED_COLUMNS = {"onset", "duration", "offset", "trial_number", "trial", "level", "run"}
MAX_SPREAD = 0.05          # seconds; (column - onset) must be this constant across rows
MIN_ROWS = 3
MIN_OFFSET = 1.0           # seconds; a column equal to onset is not another clock
MIN_STEADY_SHARE = 2 / 3   # share of a session's runs whose two clocks must agree
NOT_A_CLOCK = ("offset", "duration", "end", "stop")   # columns that follow onset by design


def events_clock_origin(events):
    """Task-clock time of the run's t=0, or None when the events carry no such clock."""
    onset = pd.to_numeric(events.get("onset"), errors="coerce")
    candidates = [name for name in PREFERRED_COLUMNS if name in events]
    candidates += [name for name in events.columns
                   if name not in candidates and name not in SKIPPED_COLUMNS
                   and not any(word in name.lower() for word in NOT_A_CLOCK)]
    for name in candidates:
        values = pd.to_numeric(events[name], errors="coerce")
        difference = (values - onset).dropna()
        if len(difference) < MIN_ROWS:
            continue
        origin = float(difference.median())
        spread = float((difference - origin).abs().median())
        if spread <= MAX_SPREAD and abs(origin) >= MIN_OFFSET:
            return origin
    return None


def time_of_day(text):
    """'13:05:42.5' -> seconds since midnight, NaN when missing or malformed."""
    try:
        hours, minutes, seconds = str(text).split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return float("nan")


def clock_shifts(runs, tr):
    """Shift of each run's BOLD start according to the clocks, in volumes.

    `runs` has columns id, sub, ses, acq_time and clock_origin. Returns a Series
    indexed like `runs`: NaN when a run has no clock evidence (no task clock in
    its events, fewer than three such runs in its session, a session in which
    the two clocks do not agree in most runs, or a task clock that was
    restarted during the session).
    """
    acquired = runs.acq_time.map(time_of_day)
    origin = pd.to_numeric(runs.clock_origin, errors="coerce")
    offset = acquired - origin
    grouped = offset.groupby([runs["sub"], runs["ses"]])
    deviation = offset - grouped.transform("median")
    enough_runs = grouped.transform("count") >= MIN_ROWS
    # a column is the task clock only if it behaves like one: in most runs of the session
    # it runs in step with the scanner's clock
    in_step = (deviation.abs() <= tr).groupby([runs["sub"], runs["ses"]]).transform("mean")
    steady = in_step >= MIN_STEADY_SHARE
    deviation[~enough_runs | ~steady | (deviation.abs() > settings.CLOCK_MAX_DEVIATION)] = np.nan
    return (-deviation / tr).round(1)
