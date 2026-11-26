"""Load one dataset for the scan: BOLD per run, predictors per run, run facts."""

from pathlib import Path

import h5py
import numpy as np

from analysis import datasets, settings, signals
from analysis.checks import clock
from analysis.predictors import events as event_predictors
from analysis.predictors import games as game_predictors


def load_bold(timeseries_root, index):
    """{run id: detrended, z-scored (volumes x parcels) array}, first volumes dropped."""
    bold = {}
    for h5_path, runs in index.groupby("h5"):
        with h5py.File(Path(timeseries_root) / h5_path) as handle:
            for run in runs.itertuples():
                data = np.asarray(handle[run.h5_key][()], float)
                if len(data) > settings.MIN_VOLUMES:
                    kept = data[settings.DROPPED_VOLUMES:]
                    bold[run.id] = signals.detrend_zscore(kept).astype(np.float32)
    return bold


def run_predictors(run, bids_root, tr):
    """(predictors, info) for one run, from its events file."""
    try:
        events = event_predictors.read_events(Path(bids_root) / run.events)
    except Exception as error:                  # unreadable events: reported, not fatal
        return {}, {"events_error": str(error)[:200]}
    if "onset" not in events or not events.onset.notna().any():
        return {}, {"events_error": "events file has no rows"}
    if game_predictors.is_game_events(events):
        predictors, info = game_predictors.build(events, int(run.n_vols), tr, bids_root)
    else:
        predictors, info = event_predictors.build(events, int(run.n_vols), tr)
    info["clock_origin"] = clock.events_clock_origin(events)
    return predictors, info


def load_dataset(checkout, dataset, index, subjects=None):
    """Everything the scan needs for one dataset.

    Returns (index, bold, predictors, info, tr, events_only): the index is
    restricted to runs that have a usable time series, and gains the columns
    `family` and `ident`. `events_only` lists the runs that have events but no
    time series: their predictors are built too, so that the pairing check can
    find a BOLD run that goes with the events of a run whose own BOLD is absent.
    """
    index = index.copy()
    if subjects:
        index = index[index["sub"].isin(subjects)]
    index["family"] = [datasets.family(dataset, task) for task in index.task]
    index["ident"] = [datasets.identity(dataset, run.task, run.ses, run.run)
                      for run in index.itertuples()]
    with_series = index[(index.h5 != "") & index.n_vols.notna()]
    tr = float(with_series.tr.median()) if len(with_series) else settings.DEFAULT_TR
    bold = load_bold(datasets.timeseries_dir(checkout, dataset), with_series)
    events_only = index[(index.h5 == "") & index.events_present].copy()
    events_only["n_vols"] = with_series.n_vols.median() if len(with_series) else np.nan
    index = with_series[with_series.id.isin(bold)].reset_index(drop=True)
    bids_root = datasets.bids_dir(checkout, dataset)
    predictors, info = {}, {}
    for run in index.itertuples():
        info[run.id] = {}
        if run.events and run.events_present:
            predictors[run.id], info[run.id] = run_predictors(run, bids_root, tr)
    for run in events_only[events_only.n_vols.notna()].itertuples():
        predictors[run.id], _ = run_predictors(run, bids_root, tr)
    return index, bold, predictors, info, tr, events_only.reset_index(drop=True)
