"""Pairing: does a run's BOLD fit its own events better than another run's?

For each predictor, margin = fit with the run's own events - best fit with the
events of another run of the same task (same or neighbouring session, not an
identical design). Margins are standardised per task and predictor (median,
MAD) and averaged over the reliable predictors: the run's pairing score.

A run is flagged when the score is low, the predictors (all but one at most)
prefer another run, and the predictors that tell runs apart best (stimulus
on/off, button presses, other subjects) agree on which one and do not prefer
the run's own events.
"""

import numpy as np
import pandas as pd

from analysis import settings, signals

FLAGGED = "events fit another run better"
FINE = "fits its own events"
MAX_SIMULATED_SWAPS = 6
PRIMARY_PREDICTORS = ("stim", "resp", "isc")


def margins(cells, task_of):
    """One row per (run, predictor): own fit, best other fit, margin, simulated swaps."""
    rows = []
    for (run_id, predictor), group in cells.groupby(["bold", "pred"]):
        own = group[group.events == run_id].r
        others = group[(group.events != run_id) & ~group.same]
        if own.empty or others.empty:
            continue
        own_fit, best = float(own.iloc[0]), others.r.idxmax()
        # a simulated swap: the k-th other run's events take the place of the run's own
        swapped = [float(fit - max(own_fit, *(others.r.drop(key).tolist() or [-1])))
                   for key, fit in others.r.items()][:MAX_SIMULATED_SWAPS]
        rows.append(dict(id=run_id, pred=predictor, task=task_of[run_id], own=own_fit,
                         best_other=float(others.r.max()), other=others.events[best],
                         margin=own_fit - float(others.r.max()), n=len(others), sim=swapped))
    return pd.DataFrame(rows)


def standardise(table):
    """Add z (margin) and zsim (simulated swaps), per task and predictor."""
    table = table.copy()
    table["z"] = np.nan
    table["zsim"] = None
    for _, group in table.groupby(["task", "pred"]):
        centre = group.margin.median()
        spread = signals.mad(group.margin.values) or 1e-3
        table.loc[group.index, "z"] = (group.margin - centre) / spread
        table.loc[group.index, "zsim"] = pd.Series(
            [[(value - centre) / spread for value in swaps] for swaps in group.sim],
            index=group.index)
    return table


def run_result(group):
    """Pairing result of one run from its reliable predictors' rows."""
    score = float(group.z.mean())
    prefer_other = int((group.margin < 0).sum())
    needed = len(group) - (len(group) >= 3)
    primary = group[group.pred.isin(PRIMARY_PREDICTORS)]
    # recording-based predictors repeat across plays of the same level, so only the
    # primary ones are asked to agree on which run fits better (all of them when
    # the run has no primary predictor)
    voters = primary if len(primary) else group
    candidate = voters.other.mode().iloc[0]
    agreeing = (voters.other == candidate).sum() >= len(voters) - (len(voters) >= 3)
    flagged = score < settings.PAIR_Z and prefer_other >= needed and agreeing \
        and not (primary.margin > 0).any()
    return dict(pair_score=round(score, 2), pair_n=len(group), pair_neg=prefer_other,
                pair_other=candidate if flagged else "",
                flag_pair=FLAGGED if flagged else FINE)


def check(cells, task_of, reliable_keys):
    """Pairing result per run id.

    `cells` is the scan's pairing table, `task_of` maps a run id to its task,
    `reliable_keys` is the set of reliable (task, predictor) pairs. Returns
    (results, margins table, sensitivity to a simulated swap in percent).
    """
    if not len(cells):
        return {}, pd.DataFrame(), None
    table = margins(cells, task_of)
    if not len(table):
        return {}, table, None
    table = table[[(task, predictor) in reliable_keys
                   for task, predictor in zip(table.task, table.pred)]]
    table = standardise(table)
    n_reliable = {task: sum(1 for key in reliable_keys if key[0] == task)
                  for task in table.task.unique()}
    results, caught = {}, []
    for run_id, group in table.groupby("id"):
        if len(group) < min(2, n_reliable[group.task.iloc[0]]):
            continue
        results[run_id] = run_result(group)
        for position in range(min(len(swaps) for swaps in group.zsim)):
            score = float(np.mean([swaps[position] for swaps in group.zsim]))
            caught.append(score < settings.PAIR_Z)
    sensitivity = round(100 * float(np.mean(caught)), 1) if caught else None
    return results, table, sensitivity
