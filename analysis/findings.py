"""The list of findings: one row per flagged run and check, in plain words.

This is the "culprit list" of the dashboard's Overview and of
`tables/flagged_runs.tsv`. A finding is `documented` when the run has a public
reference for that check in `known_issues.tsv`, otherwise `new`.
"""

import pandas as pd

from analysis.checks import drift, endrun, pairing, repetition_flags, shift

PREDICTOR_NAMES = {
    "stim": "stimulus on/off", "resp": "button presses", "isc": "other subjects",
    "cond1": "condition A", "cond2": "condition B", "cond3": "condition C",
    "luminance": "screen luminance", "flow": "optical flow", "audio": "audio envelope",
}
CHECK_OF_KNOWN = {"Start shift": "start shift", "Pairing": "pairing", "End of run": "end of run",
                  "Drift": "drift", "Repetitions": "repetitions"}
COMMON_COVERAGE = 20       # a coverage note shared by this many runs is a dataset-level note


def predictor_name(key, kind="task"):
    """Plain name of a predictor; the stimulus of a videogame dataset is the game."""
    if key == "stim" and kind == "videogame":
        return "game on/off"
    return PREDICTOR_NAMES.get(key, key)


def join_names(keys, kind="task"):
    names = [predictor_name(key, kind) for key in keys]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def shift_wording(flag, backers, kind="task"):
    """The start-shift finding, naming the predictors that see the shift."""
    keys = [key for key in str(backers or "").split(",") if key]
    if flag == "no fit to its stimulus":
        return "No fit to its stimulus (every predictor far below its usual fit)"
    if flag == "shift: clock + brain":
        return "Shift seen by the clocks and " + join_names(keys, kind)
    if flag == "shift: clock only":
        return "Shift seen by the clocks only"
    if flag == "clocks and brain disagree":
        return "Clocks and brain see different shifts"
    if flag.startswith("shift") and keys:
        return "Shift seen by " + join_names(keys, kind)
    return flag


def status_of(run, check):
    """'documented' when a public reference covers this run for this check."""
    if not run.get("known"):
        return "new"
    covered = str(run.get("known_check", "")).split(",")
    related = CHECK_OF_KNOWN[check] in covered or "any" in covered
    # a mispaired run also shows up as shifted, with odd ends and drifts
    return "documented" if related or "pairing" in covered else "new"


def per_predictor(run, reliable, kind="task"):
    """'stimulus on/off -3.9, button presses -4.1' for the predictors that count."""
    parts = []
    for column, value in run.items():
        if column.startswith("shift_") and column != "shift_vols" and pd.notna(value) \
                and column[6:] in reliable:
            parts.append(f"{predictor_name(column[6:], kind)} {value:+.1f}")
    return ", ".join(parts)


def shift_finding(run, reliable, kind="task"):
    flag = run.get("flag_shift", "")
    if not shift.is_flagged(flag):
        return None
    size = f"{run.shift_vols:+.1f} vol. " if pd.notna(run.get("shift_vols")) else ""
    clock = f"Clocks: {run.clock_shift_vols:+.1f} vol. " \
        if pd.notna(run.get("clock_shift_vols")) else ""
    counted = per_predictor(run, reliable, kind)
    return dict(check="Start shift", flag=shift_wording(flag, run.get("backers"), kind), kind=flag,
                value=run.get("shift_vols"), where=run.get("where", "") or "",
                detail=size + clock + (f"Predictors that count: {counted}." if counted else ""))


def pairing_finding(run):
    if run.get("flag_pair") != pairing.FLAGGED:
        return None
    return dict(check="Pairing", flag=pairing.FLAGGED, kind=pairing.FLAGGED,
                value=run.pair_score, where="events file or run label",
                detail=f"Fits the events of {run.pair_other} better (score {run.pair_score:.1f}).")


def end_finding(run):
    flag = run.get("flag_end")
    if flag not in endrun.FLAGGED:
        return None
    if flag == endrun.SHORT:
        detail = f"{run.n_vols} volumes, the others of this stimulus have " \
                 f"{int(run.ident_n_median)}."
    else:
        detail = f"Tail {run.tail_s:+.1f} s after the events " \
                 f"(usual for the task: {run.task_tail_median:.1f} s)."
    return dict(check="End of run", flag=flag, kind=flag, value=run.get("tail_s"), detail=detail,
                where="BOLD run" if flag == endrun.SHORT else "events file")


def drift_finding(run):
    if run.get("flag_drift") != drift.FLAGGED:
        return None
    return dict(check="Drift", flag=drift.FLAGGED, kind=drift.FLAGGED, value=run.drift_score,
                where="", detail=f"Drift score {run.drift_score:+.1f} "
                                 "(the shift changes along the run).")


def run_findings(findings):
    """Findings of the four run-level checks for one dataset."""
    runs, reliability = findings["runs"], findings["reliability"]
    rows = []
    for _, run in runs.iterrows():
        reliable = set(reliability[reliability.reliable & (reliability.task == run.task)].pred)
        found = [shift_finding(run, reliable, findings["kind"]), pairing_finding(run),
                 end_finding(run),
                 drift_finding(run)]
        for finding in found:
            if finding:
                rows.append(dict(dataset=findings["dataset"], run=run.id, task=run.task_bids,
                                 status=status_of(run, finding["check"]), known=run.known,
                                 url=run.known_url, events=run.events, bold=run.bold, **finding))
    return rows


def coverage_findings(findings):
    """(rows for runs with their own coverage problem, dataset-level note for shared ones)."""
    table = findings["coverage"]
    problems = table[table.coverage != "ok"]
    counts = problems.coverage.value_counts()
    common = counts[counts >= COMMON_COVERAGE]
    rows = [dict(dataset=findings["dataset"], run=row.id, task=row.task, status="new", known="",
                 url="", events=row.events, bold=row.bold, check="Coverage", flag=row.coverage,
                 kind=row.coverage, value=None, where="", detail="")
            for row in problems[~problems.coverage.isin(common.index)].itertuples()]
    note = "; ".join(f"{count} runs: {text}" for text, count in common.items())
    return rows, note


def repetition_findings(dataset, judged, runs):
    """Findings of the Repetitions check (videogame datasets)."""
    plays = judged["plays"]
    by_id = runs.set_index("id")
    rows = []
    for play in plays[plays.flag.isin(repetition_flags.FLAGGED)].itertuples():
        run = by_id.loc[play.run_id]
        detail = f"Slot deviation {play.slot_dev:+.1f} s." if pd.notna(play.slot_dev) else ""
        if play.swap_with:
            detail += f" Swapped with {play.swap_with}."
        rows.append(dict(dataset=dataset, run=play.id, task=run.task_bids,
                         status=status_of(run, "Repetitions"), known=run.known,
                         url=run.known_url, events=run.events, bold=run.bold,
                         check="Repetitions", flag=play.flag, kind=play.flag,
                         value=play.slot_dev, where="replay file or events row", detail=detail))
    return rows
