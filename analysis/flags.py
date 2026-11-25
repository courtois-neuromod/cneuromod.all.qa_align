"""Turn one dataset's scan into findings: one row per run, every check.

Reads the scan (analysis.scan) and the run index (analysis.bids_index) and
applies the checks of analysis.checks. Nothing here touches the BOLD again.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import datasets, known_issues, settings
from analysis.checks import clock, drift, endrun, pairing, shift

RUN_COLUMNS = ["dataset", "id", "sub", "ses", "task", "task_bids", "run", "n_vols", "tr",
               "ident", "events", "bold", "h5_key", "note"]


def fits_table(scan, index):
    """The scan's (run, predictor) fits with the task family and TR of each run."""
    fits = pd.DataFrame(scan["runs"], columns=["id", "pred", "best_lag_s", "r_best", "curve"])
    fits["task"] = fits.id.map(index.family)
    fits["tr"] = scan["tr"]
    return fits


def runs_table(scan, index):
    runs = index.reset_index().rename(columns={"task": "task_bids", "family": "task"})
    runs["dataset"] = scan["dataset"]
    runs["tr"] = scan["tr"]
    runs["n_vols"] = runs.n_vols.astype(int)
    for column in RUN_COLUMNS:
        if column not in runs:
            runs[column] = ""
    info = scan["info"]
    runs["events_end_s"] = pd.to_numeric(
        runs.id.map(lambda run_id: info.get(run_id, {}).get("events_end")), errors="coerce")
    runs["clock_origin"] = runs.id.map(lambda run_id: info.get(run_id, {}).get("clock_origin"))
    return runs


def merge_results(runs, results):
    """Add per-run result dicts ({run id: {column: value}}) as columns."""
    if not results:
        return runs
    extra = pd.DataFrame.from_dict(results, orient="index")
    return runs.merge(extra, left_on="id", right_index=True, how="left")


def confirmed_clock_shift(runs):
    """Seconds by which the events must move when clocks and brain agree on a shift."""
    confirmed = runs.flag_shift == "shift: clock + brain"
    return (runs.clock_shift_vols * runs.tr).where(confirmed)


def flag_dataset(scan, index_table):
    """All findings of one dataset.

    Returns a dict: runs (one row per run), reliability, sensitivity, fits
    (curves), pairing, drift, lags, coverage and end-of-run summary.
    """
    dataset = scan["dataset"]
    index = pd.DataFrame(scan["index"]).set_index("id")
    runs = runs_table(scan, index)
    lags = np.asarray(scan["lags"], float)
    fits = fits_table(scan, index)
    reliable = shift.reliability(fits)
    periods = {run_id: facts.get("trial_period") for run_id, facts in scan["info"].items()}
    fits, reliable = shift.annotate(fits, reliable, lags, periods)
    reliable_keys = set(zip(reliable[reliable.reliable].task, reliable[reliable.reliable].pred))

    runs["clock_shift_vols"] = clock.clock_shifts(runs, scan["tr"])
    shift_rows, shift_sensitivity = shift.check(
        fits, list(runs.id), lags, dict(zip(runs.id, runs.clock_shift_vols)))
    runs = runs.drop(columns=["clock_shift_vols"]).merge(pd.DataFrame(shift_rows), on="id")

    pair_results, pair_table, pair_sensitivity = pairing.check(
        pd.DataFrame(scan["cells"]), index.family.to_dict(), reliable_keys)
    runs = merge_results(runs, pair_results)

    runs["clock_confirmed_shift_s"] = confirmed_clock_shift(runs)
    runs = endrun.check(runs, datasets.profile(dataset).tail_min)

    drift_results, drift_table, drift_rates = drift.check(
        scan["drift"], index.family.to_dict(), reliable_keys, scan["tr"])
    runs = merge_results(runs, drift_results)

    runs = known_issues.annotate(runs, dataset)
    sensitivity = dict(start_shift=shift_sensitivity, pairing_swap=pair_sensitivity,
                       **drift_rates)
    return dict(dataset=dataset, tr=scan["tr"], kind=datasets.profile(dataset).kind, runs=runs,
                reliability=reliable, sensitivity=sensitivity, lags=lags,
                fits=fits[["id", "pred", "task", "curve", "baseline_s", "reliable"]],
                pairing=pair_table, cells=pd.DataFrame(scan["cells"]), drift=drift_table,
                end_summary=endrun.summary(runs), coverage=coverage(index_table, scan),
                labels=predictor_labels(scan))


def coverage(index_table, scan):
    """One row per run of the index with what is missing for it, 'ok' when nothing is."""
    table = index_table.copy()
    table["coverage"] = table.status.where(table.note == "", table.note)
    unavailable = (table.status == "ok") & ~table.events_present & (table.note == "")
    table.loc[unavailable, "coverage"] = \
        "events file not available (checked with the other subjects only, where possible)"
    errors = {run_id: facts["events_error"] for run_id, facts in scan["info"].items()
              if "events_error" in facts}
    unreadable = table.id.isin(errors) & (table.coverage == "ok")
    table.loc[unreadable, "coverage"] = table.id.map(errors)
    short = table.n_vols.notna() & (table.n_vols <= settings.MIN_VOLUMES) & (table.coverage == "ok")
    table.loc[short, "coverage"] = "time series too short to check"
    return table


def predictor_labels(scan):
    """{task family: {condition predictor: what it is}} from the scan's run facts."""
    index = pd.DataFrame(scan["index"])
    labels = {}
    for run_id, family in zip(index.id, index.family):
        for name, text in scan["info"].get(run_id, {}).get("labels", {}).items():
            labels.setdefault(family, {}).setdefault(name, text)
    return labels


def write_flags(findings, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(findings, handle)


def read_flags(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
