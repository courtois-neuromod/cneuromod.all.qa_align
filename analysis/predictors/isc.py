"""The "other subjects" predictor: when every subject received the same
stimulus, the average BOLD of the other subjects predicts this subject's BOLD,
parcel by parcel, with no events file involved.

Because it does not read the events, it tells an events problem (the events
predictors move, the other subjects do not) from a BOLD problem (both move).
"""

import numpy as np

from analysis import settings

MIN_SUBJECTS = 3
MIN_COVERAGE = 0.8


def build_templates(index, bold):
    """Per stimulus identity: each subject's summed BOLD and the totals.

    `index` has columns id, sub and ident; `bold` maps run id to its
    (volumes x parcels) array. Runs are cut to the identity's median length.
    """
    templates = {}
    for identity, runs in index[index.ident != ""].groupby("ident"):
        if runs["sub"].nunique() < MIN_SUBJECTS:
            continue
        length = int(np.median([len(bold[run_id]) for run_id in runs.id]))
        n_parcels = bold[runs.id.iloc[0]].shape[1]
        per_subject = {}
        for run in runs.itertuples():
            data = bold[run.id][:length]
            total, count = per_subject.get(
                run.sub, (np.zeros((length, n_parcels), np.float32), np.zeros(length, np.float32)))
            total[:len(data)] += data
            count[:len(data)] += 1
            per_subject[run.sub] = (total, count)
        templates[identity] = (per_subject,
                               sum(value[0] for value in per_subject.values()),
                               sum(value[1] for value in per_subject.values()))
    return templates


def template_for(templates, identity, subject):
    """Mean BOLD of the other subjects for one identity, or None if too sparse."""
    if identity not in templates:
        return None
    per_subject, total, count = templates[identity]
    own_total, own_count = per_subject.get(subject, (0, 0))
    others = count - own_count
    if (others > 0).mean() < MIN_COVERAGE:
        return None
    template = (total - own_total) / np.maximum(others, 1)[:, None]
    template[others == 0] = 0
    return template


def sample_template(template, sample_positions, parcels):
    """Template rows at fractional positions (linear interpolation, 0 outside)."""
    lower = np.floor(sample_positions).astype(int)
    weight = (sample_positions - lower).astype(np.float32)[..., None]
    length = len(template)
    padded = np.vstack([template[:, parcels], np.zeros((1, len(parcels)), np.float32)])
    first = np.where((lower >= 0) & (lower < length), lower, length)
    second = np.where((lower + 1 >= 0) & (lower + 1 < length), lower + 1, length)
    return padded[first] * (1 - weight) + padded[second] * weight


def shifted(template, n_volumes, tr, lags, parcels, extra_delay=None):
    """(lags x volumes x parcels): the template moved later by each lag, z-scored in time."""
    positions = np.arange(n_volumes - settings.DROPPED_VOLUMES)[None, :] \
        - np.asarray(lags, float)[:, None] / tr
    if extra_delay is not None:
        positions = positions - extra_delay[None, :] / tr
    sampled = sample_template(template, positions, parcels)
    return (sampled - sampled.mean(1, keepdims=True)) / (sampled.std(1, keepdims=True) + 1e-9)


def fit_at_zero(data, template):
    """Correlation of each parcel with the template at no shift."""
    aligned = np.zeros_like(data)
    aligned[:len(template)] = template[:len(data)]
    aligned = (aligned - aligned.mean(0)) / (aligned.std(0) + 1e-9)
    return (data * aligned).mean(0)
