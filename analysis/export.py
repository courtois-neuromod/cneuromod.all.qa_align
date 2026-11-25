"""Gather the findings of every dataset into TSV tables and the dashboard's data.

Tables (output_data/tables):
  overview.tsv       one row per dataset: what was checked, what was flagged
  flagged_runs.tsv   one row per flagged run and check (the culprit list)
  runs.tsv           one row per run with the result of every check
  predictors.tsv     one row per dataset, task and predictor: reliable or not
  repetitions.tsv    videogame datasets: one row per flagged play
  coverage.tsv       runs that could not be checked, and why
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import datasets, findings, flags, pipeline
from analysis.checks import drift, endrun, pairing, repetition_flags, repetitions, shift

RUN_TABLE_COLUMNS = [
    "dataset", "id", "sub", "ses", "task_bids", "run", "n_vols", "flag_shift", "shift_vols",
    "clock_shift_vols", "backers", "where", "flag_pair", "pair_score", "pair_other", "flag_end",
    "tail_s", "flag_drift", "drift_score", "known", "known_url", "events", "bold"]


def load_dataset(output_dir, dataset):
    """Findings of one dataset, with its Repetitions findings when it has any."""
    found = flags.read_flags(pipeline.output_path(output_dir, "flags", dataset))
    replay_path = pipeline.output_path(output_dir, "repetitions", dataset)
    found["repetitions"] = repetition_flags.judge(repetitions.read(replay_path)) \
        if replay_path.exists() else None
    return found


def column(runs, name):
    return runs[name].fillna("") if name in runs else pd.Series("", index=runs.index)


def check_counts(found):
    """How many runs each check could judge, and how many it flagged."""
    runs = found["runs"]
    shift_flag, pair_flag = column(runs, "flag_shift"), column(runs, "flag_pair")
    end_flag, drift_flag = column(runs, "flag_end"), column(runs, "flag_drift")
    judged_shift = ~shift_flag.isin(["aligned (weak brain signal)", ""]) & (runs.n_reliable > 0)
    counts = dict(
        checked_shift=int(judged_shift.sum()), shift=int(shift_flag.map(shift.is_flagged).sum()),
        checked_pair=int((pair_flag != "").sum()), pair=int((pair_flag == pairing.FLAGGED).sum()),
        checked_end=int(runs.tail_s.notna().sum()), end=int(end_flag.isin(endrun.FLAGGED).sum()),
        checked_drift=int((drift_flag != "").sum()),
        drift=int((drift_flag == drift.FLAGGED).sum()),
        checked_reps=None, reps=None)
    if found["repetitions"] is not None:
        plays = found["repetitions"]["plays"]
        counts.update(checked_reps=int(plays.slot_dev.notna().sum()),
                      reps=int(plays.flag.isin(repetition_flags.FLAGGED).sum()))
    return counts


def overview_row(found, coverage_note):
    dataset, reliability = found["dataset"], found["reliability"]
    table = found["coverage"]
    reliable = sorted({findings.predictor_name(key, found["kind"])
                       for key in reliability[reliability.reliable].pred})
    notes = [text for text in (datasets.profile(dataset).note, coverage_note) if text]
    if not reliable:
        notes.insert(0, "No brain check possible: no predictor is reliable here")
    sensitivity = found["sensitivity"]
    return dict(
        dataset=dataset, kind=found["kind"], tr=found["tr"],
        tail_min_s=datasets.profile(dataset).tail_min, runs=len(table),
        with_series=len(found["runs"]),
        bids_commit=table.bids_commit.iloc[0] if len(table) else "",
        timeseries_commit=table.timeseries_commit.iloc[0] if len(table) else "",
        predictors=", ".join(reliable), **check_counts(found),
        coverage=int((table.coverage != "ok").sum()),
        sens_shift=sensitivity.get("start_shift"), sens_pair=sensitivity.get("pairing_swap"),
        sens_jump=sensitivity.get("drift_jump"), sens_drift=sensitivity.get("drift_drift"),
        drift_rate=sensitivity.get("drift_false_alarm"), note="; ".join(notes))


def dataset_findings(found):
    """(culprit rows, dataset-level coverage note) for one dataset."""
    rows = findings.run_findings(found)
    coverage_rows, note = findings.coverage_findings(found)
    rows += coverage_rows
    if found["repetitions"] is not None:
        rows += findings.repetition_findings(found["dataset"], found["repetitions"],
                                             found["runs"])
    return rows, note


def gather(output_dir, names):
    """Everything the tables and the dashboard are made of, for the datasets given."""
    loaded, overview, culprits = {}, [], []
    for dataset in names:
        found = load_dataset(output_dir, dataset)
        rows, note = dataset_findings(found)
        loaded[dataset] = found
        overview.append(overview_row(found, note))
        culprits += rows
    overview = pd.DataFrame(overview).astype({"checked_reps": "Int64", "reps": "Int64"})
    return loaded, overview, pd.DataFrame(culprits)


def runs_table(loaded):
    frames = []
    for found in loaded.values():
        runs = found["runs"]
        keep = [name for name in RUN_TABLE_COLUMNS if name in runs]
        per_predictor = sorted(name for name in runs
                               if name.startswith(("shift_", "r_", "slope_"))
                               and name not in ("shift_vols",))
        frames.append(runs[keep + per_predictor])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_tables(output_dir, names):
    loaded, overview, culprits = gather(output_dir, names)
    folder = Path(output_dir) / pipeline.STEP_FOLDERS["tables"]
    folder.mkdir(parents=True, exist_ok=True)
    tables = {
        "overview.tsv": overview,
        "flagged_runs.tsv": culprits,
        "runs.tsv": runs_table(loaded),
        "predictors.tsv": pd.concat(
            [found["reliability"].assign(dataset=name) for name, found in loaded.items()],
            ignore_index=True) if loaded else pd.DataFrame(),
        "repetitions.tsv": pd.concat(
            [found["repetitions"]["plays"].drop(columns=["cand"]).assign(dataset=name)
             .query("flag != 'fits'") for name, found in loaded.items()
             if found["repetitions"] is not None] or [pd.DataFrame()], ignore_index=True),
        "coverage.tsv": pd.concat(
            [found["coverage"].query("coverage != 'ok'")[
                ["dataset", "id", "task", "coverage", "bold", "events", "h5_key"]]
             for found in loaded.values()] or [pd.DataFrame()], ignore_index=True),
    }
    for name, table in tables.items():
        table.to_csv(folder / name, sep="\t", index=False, na_rep="n/a")
        print(f"tables: {name} {len(table)} rows")
    return loaded, overview, culprits


def clean(value):
    """A JSON-safe value: NaN becomes None, numpy scalars become Python ones."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 3)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return None if value is pd.NA else value


def records(table):
    return [{key: clean(value) for key, value in row.items()}
            for row in table.to_dict("records")]


def dashboard_data(output_dir, names):
    """The dashboard's data (see analysis.dashboard.payload for the per-dataset part)."""
    from analysis.dashboard import payload

    loaded, overview, culprits = gather(output_dir, names)
    return payload.assemble(loaded, records(overview), records(culprits))
