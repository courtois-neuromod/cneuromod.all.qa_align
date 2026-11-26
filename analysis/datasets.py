"""What the checks need to know about each CNeuroMod dataset.

A dataset is any folder of a cneuromod.all checkout that has both a `bids` and
a `timeseries` subdataset installed. Everything else is optional knowledge: a
dataset that is not listed in `PROFILES` is checked with the defaults, so a new
dataset works without touching this file.

Three things can differ between datasets:

kind      "videogame" datasets have recordings of each play (bk2, frame-wise
          features), which give extra predictors and the Repetitions check.
          "naturalistic" and "task" only change wording on the dashboard.
family    runs that are comparable: they share parcels, the usual shift and the
          pairing candidates. The task by default; a whole season or movie for
          naturalistic sets, where each run has its own task label.
identity  runs in which every subject received the same stimulus, so the other
          subjects' BOLD can serve as a predictor ("other subjects"). None when
          the stimulus differs between subjects (videogames, shuffled orders).
tail_min  how long a scan may go on after the last event before it is unusual.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    kind: str = "task"
    family: str = "task"            # "task", "dataset", or a regex whose first group is the family
    identity: str | None = "task_run"   # "task_run", "task_session_run" or None
    tail_min: float = 15.0          # seconds; smallest unusual tail after the last event
    note: str = ""


# A run of a videogame dataset ends when the time is up, wherever the player is: the
# scan can go on for up to a play after the last event.
VIDEOGAME = Profile(kind="videogame", identity=None, tail_min=60.0)

PROFILES = {
    "mario": VIDEOGAME,
    "mariostars": VIDEOGAME,
    "mario3": VIDEOGAME,
    "shinobi": VIDEOGAME,
    "friends": Profile(kind="naturalistic", family="dataset"),
    "ood": Profile(kind="naturalistic", family="dataset"),
    "movie10": Profile(kind="naturalistic", family=r"^(.*?)\d*$"),
    "narratives": Profile(kind="naturalistic", family=r"(Story|Recall|Recency)$"),
    "petit-prince": Profile(kind="naturalistic"),
    "harrypotter": Profile(kind="naturalistic"),
    "mutemusic": Profile(kind="naturalistic", identity=None,
                         note="The music order differs between subjects"),
    "things": Profile(identity="task_session_run",
                      note="Trials come every 4.5 s (3.0 volumes): a shift of a whole number "
                           "of trials fits as well as the right one"),
    "triplets": Profile(identity="task_session_run"),
    "multfs": Profile(identity=None, note="The trial order differs between subjects"),
}

BIDS_SUBDATASET = "bids"
TIMESERIES_SUBDATASET = "timeseries"
TIMESERIES_GLOB = "timeseries/cneuromod2026/sub-*/*atlas-cneuromod26*_timeseries.h5"


def profile(dataset):
    return PROFILES.get(dataset, Profile())


def family(dataset, task):
    """The group of comparable runs that `task` belongs to."""
    rule = profile(dataset).family
    if rule == "task":
        return task
    if rule == "dataset":
        return dataset
    match = re.search(rule, task)
    return match.group(1) if match and match.group(1) else task


def identity(dataset, task, session, run):
    """Label shared by the runs in which every subject got the same stimulus."""
    rule = profile(dataset).identity
    if rule is None:
        return ""
    if rule == "task_session_run":
        return f"{task}|{session}|{run}"
    return f"{task}|{run}"


def bids_dir(checkout, dataset):
    return Path(checkout) / dataset / BIDS_SUBDATASET


def timeseries_dir(checkout, dataset):
    return Path(checkout) / dataset / TIMESERIES_SUBDATASET


def is_installed(path):
    """True when a subdataset has been installed (it then has its own .git)."""
    return (Path(path) / ".git").exists()


def available(checkout):
    """Datasets of the checkout that have both subdatasets installed."""
    names = []
    for folder in sorted(Path(checkout).iterdir()):
        if is_installed(folder / BIDS_SUBDATASET) and is_installed(folder / TIMESERIES_SUBDATASET):
            names.append(folder.name)
    return names


def declared(checkout):
    """Datasets for which cneuromod.all registers both a bids and a timeseries subdataset."""
    text = (Path(checkout) / ".gitmodules").read_text()
    paths = set(re.findall(r"^\s*path\s*=\s*(\S+)\s*$", text, flags=re.M))
    names = {path.split("/")[0] for path in paths if path.endswith("/" + TIMESERIES_SUBDATASET)}
    return sorted(name for name in names if f"{name}/{BIDS_SUBDATASET}" in paths)
