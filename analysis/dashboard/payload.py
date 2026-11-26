"""The data the dashboard page embeds, as one JSON-serialisable dict.

Large series (fit curves, pairing fits, drift windows) are stored as base64
int8 arrays with a scale, which keeps the page a few megabytes for every
dataset together. Field names of the per-run rows are short on purpose.

Top-level keys
  meta        rules (name, value, meaning), predictor names, lags (seconds)
  overview    one row per dataset (analysis.export.overview_row)
  findings    one row per flagged run and check (analysis.findings)
  runs        one compact row per run (see RUN_FIELDS)
  predictors  one row per dataset, task, predictor: baseline, reliable, ...
  curves      {dataset: {scale, runs: {run id: {predictor: b64}},
               typical: {"task|predictor": {p25, median, p75}}}}
  pairing     {dataset: {events: [run ids], scale, missing,
               cells: {predictor: {run id: [[index into events...], b64 fits]}},
               margins: [{id, pred, own, best_other, other, z}]}}
  drift       {dataset: {t0_s, dt_s, lag_scale, scale,
               windows: {predictor: {run id: [b64 shifts, b64 fits]}}}}
  end         {dataset: [{task, runs, median, p25, p75}]}
  reps        {dataset: {plays: [...], sim, stats}} for videogame datasets
  coverage    runs that could not be checked in full: {ds, id, task, coverage, events, bold}
"""

import base64

import numpy as np
import pandas as pd

from analysis import findings, settings

CURVE_SCALE = 127          # correlations are stored as round(r * 127) in int8
FIT_SCALE = 127
LAG_SCALE = 4              # drift shifts are stored as round(seconds * 4)
MISSING = -128
RUN_FIELDS = {             # short key -> column of the runs table
    "id": "id", "sub": "sub", "ses": "ses", "task": "task", "tb": "task_bids", "n": "n_vols",
    "fs": "flag_shift", "sh": "shift_vols", "bk": "backers", "cs": "clock_shift_vols",
    "wh": "where", "nr": "n_reliable", "fp": "flag_pair", "pz": "pair_score",
    "po": "pair_other", "pn": "pair_n", "pg": "pair_neg", "fe": "flag_end", "tail": "tail_s",
    "ee": "events_end_s", "fd": "flag_drift", "dz": "drift_score", "kn": "known",
    "kc": "known_check", "ku": "known_url",
}


def clean(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 3)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def encode(values, scale):
    """base64 of round(values * scale) as int8; NaN becomes MISSING."""
    array = np.asarray(values, float) * scale
    stored = np.where(np.isnan(array), MISSING, np.clip(np.round(array), -127, 127))
    return base64.b64encode(stored.astype(np.int8).tobytes()).decode()


def compact_run(dataset, run):
    row = {"ds": dataset}
    for key, name in RUN_FIELDS.items():
        value = clean(run.get(name))
        if value not in (None, ""):
            row[key] = value
    per_predictor = {}
    for name, value in run.items():
        if name.startswith("r_") and pd.notna(value):
            key = name[2:]
            per_predictor[key] = [clean(run.get(f"shift_{key}")), clean(value),
                                  clean(run.get(f"gain_{key}")), clean(run.get(f"slope_{key}"))]
    row["p"] = per_predictor
    return row


def curves(found):
    fits = found["fits"]
    data = {"scale": CURVE_SCALE, "runs": {}, "typical": {}}
    for (task, predictor), group in fits.groupby(["task", "pred"]):
        matrix = np.array([np.asarray(curve, float) for curve in group.curve])
        low, middle, high = np.percentile(matrix, [25, 50, 75], axis=0)
        data["typical"][f"{task}|{predictor}"] = {
            "p25": encode(low, CURVE_SCALE), "median": encode(middle, CURVE_SCALE),
            "p75": encode(high, CURVE_SCALE), "runs": len(group)}
        for run_id, curve in zip(group.id, group.curve):
            data["runs"].setdefault(run_id, {})[predictor] = encode(curve, CURVE_SCALE)
    return data


def pairing(found):
    cells, margins = found["cells"], found["pairing"]
    data = {"events": [], "scale": FIT_SCALE, "missing": MISSING, "cells": {}, "margins": []}
    if not len(cells):
        return data
    position = {run_id: index for index, run_id in enumerate(sorted(set(cells.events)))}
    data["events"] = list(position)
    for (predictor, run_id), group in cells.groupby(["pred", "bold"]):
        data["cells"].setdefault(predictor, {})[run_id] = [
            [position[other] for other in group.events], encode(group.r.values, FIT_SCALE)]
    if len(margins):
        keep = margins[["id", "pred", "own", "best_other", "other", "z"]]
        data["margins"] = [{key: clean(value) for key, value in row.items()}
                           for row in keep.to_dict("records")]
    return data


def drift_windows(found):
    table = found["drift"]
    step = settings.DRIFT_STEP * found["tr"]
    first = (settings.DROPPED_VOLUMES + settings.DRIFT_WINDOW / 2) * found["tr"]
    data = {"t0_s": round(first, 2), "dt_s": round(step, 2), "lag_scale": LAG_SCALE,
            "scale": FIT_SCALE, "windows": {}}
    if not len(table):
        return data
    for row in table[table.variant == "real"].itertuples():
        windows = np.array(row.windows)
        data["windows"].setdefault(row.pred, {})[row.id] = [
            encode(windows[:, 1], LAG_SCALE), encode(windows[:, 2], FIT_SCALE)]
    return data


def repetitions(found):
    judged = found["repetitions"]
    if judged is None:
        return None
    plays = []
    for play in judged["plays"].itertuples():
        row = {"id": play.id, "run": play.run_id, "lv": play.level, "dev": clean(play.slot_dev)}
        if play.flag != "fits":
            row["flag"] = play.flag
        if play.swap_with:
            row["swap"] = play.swap_with
        if play.cand:
            row["cand"] = [[candidate["stim"], clean(candidate["slot_dev"]), clean(candidate["z"])]
                           for candidate in play.cand]
        plays.append(row)
    return {"plays": plays, "sim": judged["sim"], "stats": judged["stats"]}


def predictor_rows(found):
    rows = []
    labels = found["labels"]
    for row in found["reliability"].to_dict("records"):
        row = {key: clean(value) if not isinstance(value, list) else value
               for key, value in row.items()}
        row["dataset"] = found["dataset"]
        row["label"] = labels.get(row["task"], {}).get(row["pred"], "")
        rows.append(row)
    return rows


def coverage_rows(found):
    table = found["coverage"]
    return [dict(ds=found["dataset"], id=row.id, task=row.task, coverage=row.coverage,
                 events=row.events, bold=row.bold)
            for row in table[table.coverage != "ok"].itertuples()]


def assemble(loaded, overview, culprits):
    """The whole payload from the loaded findings of each dataset."""
    data = dict(
        meta=dict(
            rules=[dict(name=name, value=value, meaning=meaning)
                   for name, value, meaning in settings.RULES],
            predictor_names=findings.PREDICTOR_NAMES,
            lags_s=[float(lag) for lag in settings.LAGS[::settings.CURVE_STEP]]),
        overview=overview, findings=culprits, runs=[], predictors=[], curves={}, pairing={},
        drift={}, end={}, reps={}, coverage=[])
    for dataset, found in loaded.items():
        data["runs"] += [compact_run(dataset, run) for _, run in found["runs"].iterrows()]
        data["predictors"] += predictor_rows(found)
        data["curves"][dataset] = curves(found)
        data["pairing"][dataset] = pairing(found)
        data["drift"][dataset] = drift_windows(found)
        data["end"][dataset] = found["end_summary"]
        data["coverage"] += coverage_rows(found)
        replays = repetitions(found)
        if replays is not None:
            data["reps"][dataset] = replays
    return data
