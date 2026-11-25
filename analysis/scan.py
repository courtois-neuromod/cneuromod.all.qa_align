"""The brain-signal scan of one dataset.

For every run and every predictor it has:

shift    the fit between predictor and BOLD when the predictor is moved by
         each lag of settings.LAGS. The fit is the mean correlation over the 50
         parcels that follow that predictor best at no shift in the subject's
         OTHER runs of the same family (leave-one-run-out).
pairing  the same fit, with the predictors of the subject's other runs of the
         family in the same and the neighbouring sessions (including runs that
         have events but no time series).
drift    the best shift in sliding windows along the run, searched around the
         run's own best shift; also with a simulated jump and a simulated
         drift in a sample of runs, to measure what the check can see.

Nothing is flagged here: `analysis.flags` turns these numbers into findings.
"""

import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import loading, settings, signals
from analysis.predictors import isc

STATE = {}                 # shared with the worker processes (inherited on fork)
ISC = "isc"
IDENTICAL_DESIGN_R = 0.9   # two runs whose predictors correlate above this are the same design


class EventSignal:
    """A predictor defined on the time grid (events, recordings)."""

    def __init__(self, signal, n_volumes, tr):
        self.signal, self.n_volumes, self.tr = signal, n_volumes, tr

    def regressors(self, lags, parcels=None, extra_delay=None, n_volumes=None):
        return signals.lagged_regressors(self.signal, n_volumes or self.n_volumes, self.tr,
                                         lags, extra_delay)

    def fit_at_zero(self, data):
        return data.T @ self.regressors([0.0])[0] / len(data)


class OtherSubjectsSignal:
    """The other subjects' BOLD for the same stimulus (one series per parcel)."""

    def __init__(self, template, n_volumes, tr):
        self.template, self.n_volumes, self.tr = template, n_volumes, tr

    def regressors(self, lags, parcels=None, extra_delay=None, n_volumes=None):
        return isc.shifted(self.template, n_volumes or self.n_volumes, self.tr, lags, parcels,
                           extra_delay)

    def fit_at_zero(self, data):
        return isc.fit_at_zero(data, self.template)


def correlate(regressors, data):
    """Mean correlation over parcels, one value per lag."""
    if regressors.ndim == 2:
        return (signals.zscore_rows(regressors) @ data).mean(1) / len(data)
    centred = regressors - regressors.mean(1, keepdims=True)
    scaled = centred / (centred.std(1, keepdims=True) + 1e-9)
    return np.einsum("ltp,tp->l", scaled, data) / data.size


def family_signals(runs, predictor, subject):
    """{run id: signal object or None} for one predictor in one family of runs."""
    found = {}
    for run in runs.itertuples():
        if run.id not in STATE["bold"]:          # events only: a pairing candidate, no BOLD
            signal = STATE["predictors"].get(run.id, {}).get(predictor)
            found[run.id] = None if signal is None or predictor == ISC else \
                EventSignal(signal, int(run.n_vols), STATE["tr"])
            continue
        n_volumes, tr = len(STATE["bold"][run.id]) + settings.DROPPED_VOLUMES, STATE["tr"]
        if predictor == ISC:
            template = isc.template_for(STATE["templates"], run.ident, subject)
            found[run.id] = None if template is None else \
                OtherSubjectsSignal(template, n_volumes, tr)
        else:
            signal = STATE["predictors"].get(run.id, {}).get(predictor)
            found[run.id] = None if signal is None else EventSignal(signal, n_volumes, tr)
    return found


def top_parcels(zero_fits, run_id):
    """Parcels that follow the predictor best in the other runs (leave this one out)."""
    others = (sum(zero_fits.values()) - zero_fits[run_id]) / (len(zero_fits) - 1)
    return np.argsort(others)[-settings.N_TOP_PARCELS:]


def pairing_cells(run_id, predictor, candidates, found, runs_by_id, sessions, parcels):
    """Fit of this run's BOLD with the predictor of each nearby run of the family."""
    data = STATE["bold"][run_id][:, parcels]
    n_volumes = len(data) + settings.DROPPED_VOLUMES
    position = sessions.index(runs_by_id.ses[run_id])
    nearby = set(sessions[max(0, position - 1): position + 2])
    lags = settings.LAGS[::settings.PAIRING_LAG_STEP]
    own_at_zero = None if predictor == ISC else \
        found[run_id].regressors([0.0], n_volumes=n_volumes)[0]
    cells = []
    for other in candidates:
        if runs_by_id.ses[other] not in nearby:
            continue
        fit = correlate(found[other].regressors(lags, parcels, n_volumes=n_volumes), data)
        if predictor == ISC:
            same = runs_by_id.ident[other] == runs_by_id.ident[run_id]
        else:
            other_at_zero = found[other].regressors([0.0], n_volumes=n_volumes)[0]
            same = other != run_id and \
                float(own_at_zero @ other_at_zero / len(own_at_zero)) > IDENTICAL_DESIGN_R
        cells.append(dict(pred=predictor, bold=run_id, events=other,
                          r=round(float(fit.max()), 4), same=bool(same)))
    return cells


def simulated_delays(n_volumes, tr):
    """Time-varying delays (seconds per volume) for the simulated jump and drift."""
    times = signals.volume_times(n_volumes, tr)
    size = settings.SIMULATED_VOLUMES * tr
    return {"jump": np.where(times >= times[len(times) // 2], size, 0.0),
            "drift": size * (times - times[0]) / (times[-1] - times[0])}


def drift_windows(signal, data, parcels, best_lag, extra_delay):
    """[window centre (s), best shift around the run's own (s), fit] per sliding window."""
    lags = best_lag + settings.DRIFT_SEARCH
    regressors = signal.regressors(lags, parcels, extra_delay)
    windows = []
    for start in range(0, len(data) - settings.DRIFT_WINDOW + 1, settings.DRIFT_STEP):
        stop = start + settings.DRIFT_WINDOW
        fit = correlate(regressors[:, start:stop], signals.detrend_zscore(data[start:stop]))
        best = int(fit.argmax())
        centre = (start + settings.DROPPED_VOLUMES + settings.DRIFT_WINDOW / 2) * signal.tr
        windows.append([round(centre, 1), round(float(settings.DRIFT_SEARCH[best]), 2),
                        round(float(fit[best]), 3)])
    return windows


def run_drift(run_id, predictor, signal, parcels, best_lag):
    data = STATE["bold"][run_id][:, parcels].astype(float)
    if len(data) < settings.DRIFT_WINDOW + 7 * settings.DRIFT_STEP:
        return []
    variants = {"real": None}
    if run_id in STATE["simulated"]:
        variants.update(simulated_delays(signal.n_volumes, signal.tr))
    return [dict(id=run_id, pred=predictor, variant=name, best_lag_s=best_lag,
                 windows=drift_windows(signal, data, parcels, best_lag, delay))
            for name, delay in variants.items()]


def scan_predictor(runs, predictor, subject, sessions, output):
    """Shift curve, pairing cells and drift windows of each run for one predictor."""
    found = family_signals(runs, predictor, subject)
    candidates = [run_id for run_id in runs.id if found[run_id] is not None]
    usable = [run_id for run_id in candidates if run_id in STATE["bold"]]
    if len(usable) < settings.MIN_RUNS_PER_PREDICTOR:
        return
    zero_fits = {run_id: found[run_id].fit_at_zero(STATE["bold"][run_id]) for run_id in usable}
    runs_by_id = runs.set_index("id")
    for run_id in usable:
        parcels = top_parcels(zero_fits, run_id)
        data = STATE["bold"][run_id][:, parcels]
        curve = correlate(found[run_id].regressors(settings.LAGS, parcels), data)
        best_lag = float(settings.LAGS[curve.argmax()])
        output["runs"].append(dict(
            id=run_id, pred=predictor, best_lag_s=best_lag, r_best=float(curve.max()),
            curve=np.round(curve[::settings.CURVE_STEP], 4).astype(np.float32)))
        output["cells"] += pairing_cells(run_id, predictor, candidates, found, runs_by_id,
                                         sessions, parcels)
        output["drift"] += run_drift(run_id, predictor, found[run_id], parcels, best_lag)


def scan_subject(subject):
    """All predictors of all families for one subject (runs in a worker process)."""
    index = pd.concat([STATE["index"], STATE["events_only"]], ignore_index=True)
    runs = index[index["sub"] == subject]
    sessions = sorted(runs.ses.unique())
    output = dict(runs=[], cells=[], drift=[])
    for _, family_runs in runs.groupby("family"):
        names = sorted({name for run_id in family_runs.id
                        for name in STATE["predictors"].get(run_id, {})})
        for predictor in names + [ISC]:
            scan_predictor(family_runs, predictor, subject, sessions, output)
    return output


def scan_dataset(checkout, dataset, index, subjects=None, n_jobs=6):
    """Run the scan and return its results (see module docstring)."""
    index, bold, predictors, info, tr, events_only = loading.load_dataset(
        checkout, dataset, index, subjects)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    n_simulated = min(settings.N_SIMULATED_RUNS, len(index))
    simulated = set(rng.choice(index.id.values, n_simulated, replace=False)) if len(index) \
        else set()
    STATE.update(index=index, bold=bold, predictors=predictors, tr=tr, simulated=simulated,
                 events_only=events_only, templates=isc.build_templates(index, bold))
    results = dict(runs=[], cells=[], drift=[])
    subject_list = sorted(index["sub"].unique())
    if subject_list:
        with Pool(max(1, min(n_jobs, len(subject_list)))) as pool:
            for output in pool.imap_unordered(scan_subject, subject_list):
                for key in results:
                    results[key] += output[key]
    results.update(dataset=dataset, tr=tr, info=info, index=index.to_dict("records"),
                   simulated=sorted(simulated), lags=settings.LAGS[::settings.CURVE_STEP],
                   n_identities=len(STATE["templates"]))
    STATE.clear()
    return results


def write_scan(results, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)


def read_scan(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
