"""Predictors built from a run's events file alone (works for any dataset).

stim      on while a stimulus is shown: every row except responses and
          fixation/rest rows. A naturalistic run usually has one such row.
cond1-3   the (up to) three conditions that cover the most stimulus time.
resp      button presses: response rows, or onset + reaction time.

All are convolved with the HRF on the common predictor grid.
"""

import re

import numpy as np
import pandas as pd

from analysis import signals

RESPONSE_ROW = re.compile(r"^(response|keypress|key_press|press|button)", re.I)
NOT_A_STIMULUS = re.compile(r"^(fix|fixation|cross_fixation|rest|baseline|iti|blank|\+)$", re.I)
REACTION_TIME_COLUMNS = ["response_time", "reaction_time", "rt", "key_press_rt"]
CONDITION_COLUMNS = ["category", "condition", "block_type", "stim_type", "trial_type"]
DEFAULT_DURATION = 0.5
MIN_DURATION = 0.1
MIN_RESPONSES = 3
MIN_CONDITION_SHARE = 0.1
MAX_CONDITIONS = 3


def read_events(path):
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig")


def clean_events(events, run_length):
    """Rows with a numeric onset, plus a usable duration for each (`dur`)."""
    events = events[pd.to_numeric(events.get("onset"), errors="coerce").notna()].copy()
    events["onset"] = events.onset.astype(float)
    if "duration" in events:
        duration = pd.to_numeric(events.duration, errors="coerce")
    else:
        duration = pd.Series(np.nan, index=events.index)
    if "offset" in events:
        duration = duration.fillna(pd.to_numeric(events.offset, errors="coerce") - events.onset)
    if len(events) <= 2:
        # naturalistic run: the stimulus lasts until the end when no duration is given
        duration = duration.fillna(run_length - events.onset)
    events["dur"] = duration.fillna(DEFAULT_DURATION).clip(lower=MIN_DURATION)
    return events


def row_kinds(events):
    """Boolean masks (is a response row, is not a stimulus row)."""
    if "trial_type" in events:
        trial_type = events.trial_type.astype(str)
    else:
        trial_type = pd.Series("stim", index=events.index)
    is_response = trial_type.str.match(RESPONSE_ROW)
    not_stimulus = trial_type.str.match(NOT_A_STIMULUS)
    if "word" in events:
        not_stimulus = not_stimulus | (events["word"].astype(str).str.strip() == "+")
    return is_response, not_stimulus


def condition_predictors(stimuli, n_samples):
    """Boxcars of the main conditions, from the first column that has several levels."""
    for column in CONDITION_COLUMNS:
        if column not in stimuli or not stimuli[column].notna().any():
            continue
        levels = stimuli.groupby(stimuli[column].astype(str)).dur.sum()
        levels = levels[(levels.index != "nan")
                        & (levels >= MIN_CONDITION_SHARE * stimuli.dur.sum())]
        if len(levels) < 2:
            continue
        found, labels = {}, {}
        ranked = levels.sort_values(ascending=False).head(MAX_CONDITIONS)
        for position, name in enumerate(ranked.index, start=1):
            rows = stimuli[stimuli[column].astype(str) == name]
            found[f"cond{position}"] = signals.convolve_hrf(
                signals.boxcar(rows.onset, rows.dur, n_samples))
            labels[f"cond{position}"] = f"{column}={name}"
        return found, labels
    return {}, {}


def response_times(events, is_response):
    """Times of the button presses, from response rows or reaction-time columns."""
    times = list(events[is_response].onset)
    for column in REACTION_TIME_COLUMNS:
        if column not in events:
            continue
        reaction = pd.to_numeric(events[column], errors="coerce")
        valid = reaction.notna() & (reaction > 0) & ~is_response
        if valid.any():
            in_milliseconds = reaction[valid].median() > 50
            times += list(events.onset[valid] + reaction[valid] / (1000 if in_milliseconds else 1))
        break
    return times


def trial_period(stimuli):
    """Median time between stimulus onsets when trials come at a regular rhythm, else None."""
    gaps = np.diff(np.sort(stimuli.onset.values))
    gaps = gaps[gaps > 0.2]
    if len(gaps) >= 10 and np.median(gaps) < 15 and np.std(gaps) / np.median(gaps) < 0.25:
        return float(np.median(gaps))
    return None


def build(events, n_volumes, tr):
    """Predictors and facts about one run.

    Returns (predictors, info): predictors maps a name to its convolved signal;
    info holds events_start, events_end, trial_period (seconds) and the label of
    each condition predictor.
    """
    n_samples = signals.grid_length(n_volumes, tr)
    events = clean_events(events, n_volumes * tr)
    is_response, not_stimulus = row_kinds(events)
    stimuli = events[~is_response & ~not_stimulus]
    predictors, info = {}, {}
    if len(stimuli):
        predictors["stim"] = signals.convolve_hrf(
            signals.boxcar(stimuli.onset, stimuli.dur, n_samples))
        info["events_start"] = float(stimuli.onset.min())
        info["events_end"] = float((stimuli.onset + stimuli.dur).max())
        info["trial_period"] = trial_period(stimuli)
        conditions, labels = condition_predictors(stimuli, n_samples)
        predictors.update(conditions)
        info["labels"] = labels
    presses = response_times(events, is_response)
    if len(presses) >= MIN_RESPONSES:
        predictors["resp"] = signals.convolve_hrf(signals.impulses(presses, n_samples))
    return predictors, info
