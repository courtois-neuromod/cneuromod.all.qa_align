"""Signal helpers: the HRF, the predictor time grid, and lagged regressors."""

import math

import numpy as np

from analysis import settings


def canonical_hrf(step=settings.GRID_STEP):
    """Double-gamma haemodynamic response sampled every `step` seconds, sum 1."""
    time = np.arange(0, settings.HRF_LENGTH, step)

    def gamma_density(shape):
        return time ** (shape - 1) * np.exp(-time) / math.gamma(shape)

    response = gamma_density(6) - gamma_density(16) / 6
    return response / response.sum()


HRF = canonical_hrf()


def grid_length(n_volumes, tr):
    """Number of grid samples covering one run plus the padding on both sides."""
    return int(round((n_volumes * tr + settings.GRID_PAD - settings.GRID_START)
                     / settings.GRID_STEP))


def grid_index(seconds):
    """Grid sample holding the time `seconds` (may fall outside the grid)."""
    return int(round((seconds - settings.GRID_START) / settings.GRID_STEP))


def grid_times(n_samples):
    return settings.GRID_START + settings.GRID_STEP * np.arange(n_samples)


def boxcar(onsets, durations, n_samples):
    """1 while any event is on, 0 elsewhere, on the predictor grid."""
    signal = np.zeros(n_samples)
    for onset, duration in zip(onsets, durations):
        start, stop = grid_index(onset), grid_index(onset + duration)
        signal[max(start, 0):max(min(stop, n_samples), 0)] = 1
    return signal


def impulses(times, n_samples):
    """One unit-area impulse per event time (used for button presses)."""
    signal = np.zeros(n_samples)
    for time in times:
        index = grid_index(time)
        if 0 <= index < n_samples:
            signal[index] += 1 / settings.GRID_STEP / 10
    return signal


def convolve_hrf(signal):
    return np.convolve(signal, HRF)[:len(signal)].astype(np.float32)


def detrend_zscore(data):
    """Remove a quadratic trend from each column, then z-score it."""
    index = np.arange(len(data), dtype=float)
    design = np.c_[np.ones_like(index), index, index ** 2]
    residual = data - design @ np.linalg.lstsq(design, data, rcond=None)[0]
    return (residual - residual.mean(0)) / (residual.std(0) + 1e-9)


def zscore_rows(matrix):
    matrix = matrix - matrix.mean(-1, keepdims=True)
    return matrix / (matrix.std(-1, keepdims=True) + 1e-9)


def volume_times(n_volumes, tr):
    """Acquisition time of each kept volume (the first ones are dropped)."""
    return np.arange(settings.DROPPED_VOLUMES, n_volumes) * tr


def lagged_regressors(signal, n_volumes, tr, lags, extra_delay=None):
    """The predictor sampled at the volume times, once per lag (rows), z-scored.

    A positive lag moves the predictor later. `extra_delay` (seconds, one value
    per volume) adds a time-varying delay, used to simulate jumps and drifts.
    """
    times = volume_times(n_volumes, tr)[None, :] - np.asarray(lags, float)[:, None]
    if extra_delay is not None:
        times = times - extra_delay[None, :]
    return zscore_rows(np.interp(times, grid_times(len(signal)), signal))


def mad(values):
    """Median absolute deviation scaled to a standard deviation."""
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if not len(values):
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))
