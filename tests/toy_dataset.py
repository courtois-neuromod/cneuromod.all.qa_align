"""A tiny synthetic cneuromod.all checkout, to test the whole pipeline in seconds.

One dataset ("toy"), three subjects, six runs each of a block-design task
whose blocks fall at different times in every run. The BOLD of each run is its
own block predictor (convolved with the HRF) in 40 of 60 parcels, plus noise.
Two faults are planted in sub-01:

  run 2    the BOLD starts 6 volumes before the events' t=0 (a shift of +6 volumes)
  run 4/5  the events files of the two runs are exchanged
"""

import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analysis import settings, signals

TR = settings.DEFAULT_TR
N_VOLUMES = 300
N_PARCELS = 60
N_ACTIVE = 40
SUBJECTS = ("01", "02", "03")
RUNS = (1, 2, 3, 4, 5, 6)
SHIFTED = ("01", 2)
SHIFT_VOLUMES = 6
SWAPPED = ("01", 4, 5)
H5_NAME = "sub-{sub}_task-toy_space-MNI152NLin2009cAsym_atlas-cneuromod26_" \
          "desc-1134Parcels_timeseries.h5"


def block_onsets(rng):
    """Onsets of 10 blocks of 12 s, spread unevenly over the run."""
    gaps = rng.uniform(18, 50, size=10)
    return np.cumsum(gaps) - gaps[0] + 8


def events_table(onsets):
    return pd.DataFrame({"onset": np.round(onsets, 3), "duration": 12.0, "trial_type": "block"})


def bold_from(onsets, rng, early_volumes=0):
    """BOLD that follows the blocks, starting `early_volumes` before the events' t=0."""
    n_samples = signals.grid_length(N_VOLUMES, TR)
    predictor = signals.convolve_hrf(signals.boxcar(onsets, [12.0] * len(onsets), n_samples))
    times = (np.arange(N_VOLUMES) - early_volumes) * TR
    response = np.interp(times, signals.grid_times(n_samples), predictor)
    response = (response - response.mean()) / response.std()
    data = rng.normal(size=(N_VOLUMES, N_PARCELS))
    data[:, :N_ACTIVE] += 1.5 * response[:, None]
    return data.astype(np.float32)


def init_repository(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=test", "-c", "user.email=t@e.st",
                    "commit", "-q", "-m", "toy data"], check=True)


def write_bids_run(bids, subject, run, events):
    folder = bids / f"sub-{subject}" / "ses-001" / "func"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{subject}_ses-001_task-toy_run-{run:02d}"
    events.to_csv(folder / f"{stem}_events.tsv", sep="\t", index=False)
    (folder / f"{stem}_bold.nii.gz").write_bytes(b"")
    (folder / f"{stem}_bold.json").write_text(
        '{"RepetitionTime": %s, "AcquisitionTime": "10:%02d:00.000000"}' % (TR, run * 8))


def build(root, seed=0):
    """Write the toy checkout under `root` and return its path."""
    rng = np.random.default_rng(seed)
    checkout = Path(root) / "cneuromod.all"
    bids, series = checkout / "toy" / "bids", checkout / "toy" / "timeseries"
    for subject in SUBJECTS:
        folder = series / "timeseries" / "cneuromod2026" / f"sub-{subject}"
        folder.mkdir(parents=True)
        onsets = {run: block_onsets(rng) for run in RUNS}
        with h5py.File(folder / H5_NAME.format(sub=subject), "w") as handle:
            for run in RUNS:
                early = SHIFT_VOLUMES if (subject, run) == SHIFTED else 0
                handle[f"ses-001/ses-001_task-toy_run-{run}_timeseries"] = \
                    bold_from(onsets[run], rng, early)
        if subject == SWAPPED[0]:
            onsets[SWAPPED[1]], onsets[SWAPPED[2]] = onsets[SWAPPED[2]], onsets[SWAPPED[1]]
        for run in RUNS:
            write_bids_run(bids, subject, run, events_table(onsets[run]))
    init_repository(bids)
    init_repository(series)
    (checkout / ".gitmodules").write_text(
        '[submodule "toy/bids"]\n\tpath = toy/bids\n'
        '[submodule "toy/timeseries"]\n\tpath = toy/timeseries\n')
    return checkout
