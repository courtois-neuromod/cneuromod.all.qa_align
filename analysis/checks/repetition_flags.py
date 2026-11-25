"""Findings of the Repetitions check, from the measures of
`analysis.checks.repetitions`.

A play is flagged when its replay does not fill its slot in the events, or
when another replay of the same level in the session fills the slot as well
and fits the brain better, with the predictors' consensus. Two plays whose
replays each fill the other's slot are reported as swapped. A session where
most plays miss their slot has irregular gaps (another task version), not
misattributed replays.

Validation: every (play, other replay of the same level) pair is a simulated
swap; the share the rules would catch is reported.
"""

import re

import numpy as np
import pandas as pd

from analysis import settings, signals
from analysis.checks.repetitions import BRAIN_PREDICTORS

FITS = "fits"
BAD_SLOT = "replay does not fit its slot"
BRAIN = "another replay fits the brain better"
SWAPPED = "replays swapped between two plays"
IRREGULAR = "irregular gaps in the whole session"
FLAGGED = (BAD_SLOT, BRAIN, SWAPPED)
MIN_PREDICTORS = 3


def spreads(plays):
    """Per predictor: spread (MAD) of the fit differences over all candidates."""
    found = {}
    for name in BRAIN_PREDICTORS:
        values = [candidate["d"][name] for play in plays for candidate in play["cand"]
                  if candidate["d"].get(name) is not None]
        found[name] = signals.mad(values) if values else np.nan
    return found


def brain_evidence(differences, spread, sign=1):
    """(median z, flagged): does the candidate fit the brain better than the play's own replay?"""
    scores = {name: sign * differences[name] / spread[name] for name in BRAIN_PREDICTORS
              if differences.get(name) is not None and spread[name] > 0}
    if len(scores) < MIN_PREDICTORS:
        return None, False
    values = np.array(list(scores.values()))
    median, positive = float(np.median(values)), int((values > 0).sum())
    flagged = median >= settings.REPLAY_Z and positive >= len(values) - 1 \
        and scores.get("resp", 0) >= 0
    return median, flagged


def fills_slot(deviation):
    return deviation is None or abs(deviation) <= settings.SLOT_TOLERANCE


def play_label(stim, position):
    """'w1l1 rep-000' from a replay's file name, or the play's place in its run."""
    match = re.search(r"level-([^_]+)_rep-(\d+)", stim or "")
    if not match:
        return f"play {position} (no replay)"
    level = match.group(1)
    return f"{'level ' + level if level.isdigit() else level} rep-{match.group(2)}"


def judge_play(play, spread):
    """One findings row for a play (before swaps and irregular sessions)."""
    candidates = []
    for candidate in play["cand"]:
        score, hit = brain_evidence(candidate["d"], spread)
        candidates.append(dict(stim=candidate["stim"], slot_dev=candidate["slot_dev"],
                               z=None if score is None else round(score, 2),
                               hit=hit and fills_slot(candidate["slot_dev"])))
    misses_slot = play["slot_dev"] is not None and abs(play["slot_dev"]) > settings.SLOT_TOLERANCE
    if misses_slot:
        flag = BAD_SLOT
    elif any(candidate["hit"] for candidate in candidates):
        flag = BRAIN
    else:
        flag = FITS
    return dict(id=f"{play['run_id']} {play_label(play['stim'], play['play'])}",
                run_id=play["run_id"], sub=play["sub"], ses=play["ses"], run=play["run"],
                play=play["play"], level=play["level"], stim=play["stim"],
                slot_dev=play["slot_dev"], flag=flag, cand=candidates)


def find_swaps(rows):
    """{play id: id of the play it is swapped with}: each replay fills the other's slot."""
    by_stim = {(row["sub"], row["ses"], row["stim"]): row for row in rows}
    swapped = {}
    for row in rows:
        if row["flag"] != BAD_SLOT:
            continue
        for candidate in row["cand"]:
            other = by_stim.get((row["sub"], row["ses"], candidate["stim"]))
            if other is None or candidate["slot_dev"] is None \
                    or abs(candidate["slot_dev"]) > settings.SWAP_TOLERANCE:
                continue
            back = [c for c in other["cand"] if c["stim"] == row["stim"]
                    and c["slot_dev"] is not None
                    and abs(c["slot_dev"]) <= settings.SWAP_TOLERANCE]
            if back:
                swapped[row["id"]] = other["id"]
    return swapped


def mark_irregular_sessions(table):
    """Sessions where most plays miss their slot: irregular gaps, not wrong replays."""
    measured = table[table.slot_dev.notna()]
    share = measured.groupby(["sub", "ses"]).flag.apply(lambda flags: (flags == BAD_SLOT).mean())
    irregular = set(share[share > 0.5].index)
    in_irregular = np.array([(sub, ses) in irregular for sub, ses in zip(table["sub"], table.ses)])
    affected = in_irregular & table.flag.isin([BAD_SLOT, SWAPPED])
    table.loc[affected, "flag"] = IRREGULAR
    table.loc[affected, "swap_with"] = ""
    return table


def simulation(plays, spread):
    """What share of simulated swaps the rules catch, and the brain rule's false alarms."""
    n_swaps = by_slot = caught = same_length = by_brain = 0
    for play in plays:
        for candidate in play["cand"]:
            n_swaps += 1
            misses = candidate["slot_dev"] is not None \
                and abs(candidate["slot_dev"]) > settings.SLOT_TOLERANCE
            brain_hit = fills_slot(play["slot_dev"]) \
                and brain_evidence(candidate["d"], spread, sign=-1)[1]
            by_slot += misses
            caught += misses or brain_hit
            if not misses:
                same_length += 1
                by_brain += brain_hit
    with_candidates = [play for play in plays if play["cand"]]
    false_alarms = sum(any(fills_slot(c["slot_dev"]) and brain_evidence(c["d"], spread)[1]
                           for c in play["cand"]) for play in with_candidates)
    percent = lambda part, whole: round(100 * part / max(whole, 1))   # noqa: E731
    return dict(n_swaps=n_swaps, caught=percent(caught, n_swaps),
                by_slot=percent(by_slot, n_swaps),
                brain_when_same_length=percent(by_brain, same_length),
                brain_false_alarm=round(100 * false_alarms / max(len(with_candidates), 1), 2))


def judge(measures):
    """Findings for one dataset: dict(plays table, sim, stats)."""
    plays = measures["plays"]
    spread = spreads(plays)
    rows = [judge_play(play, spread) for play in plays]
    table = pd.DataFrame(rows, columns=["id", "run_id", "sub", "ses", "run", "play", "level",
                                        "stim", "slot_dev", "flag", "cand"])
    table["swap_with"] = table.id.map(find_swaps(rows)).fillna("")
    table.loc[table.swap_with != "", "flag"] = SWAPPED
    table = mark_irregular_sessions(table)
    fitting = table[table.slot_dev.notna() & (table.flag == FITS)].slot_dev.astype(float)
    stats = dict(plays=len(table), with_cand=int((table.cand.str.len() > 0).sum()),
                 p1=round(float(fitting.quantile(.01)), 2) if len(fitting) else None,
                 p99=round(float(fitting.quantile(.99)), 2) if len(fitting) else None)
    return dict(dataset=measures["dataset"], plays=table, sim=simulation(plays, spread),
                stats=stats)
