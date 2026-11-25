"""Run index of one dataset: every run of the parcel time series matched with
its BIDS BOLD file and its events file.

One row per run found in either place, with a coverage status saying what is
missing. Events files follow BIDS inheritance: the closest file whose entities
are a subset of the BOLD file's wins, and a file with content wins over an
empty template.
"""

import json
import re
import subprocess
from pathlib import Path

import h5py
import pandas as pd

from analysis import datasets, settings

ENTITY = re.compile(r"(sub|ses|task|acq|run|echo|part|dir|rec)-([A-Za-z0-9]+)")
IGNORED_FOR_EVENTS = ("part", "echo")
COLUMNS = ["dataset", "sub", "ses", "task", "acq", "run", "id", "n_vols", "tr", "acq_time",
           "h5", "h5_key", "bold", "events", "events_present", "status", "note", "bids_commit",
           "timeseries_commit"]


def entities(name):
    return dict(ENTITY.findall(name))


def number(value):
    """'002' and '02' are the same session or run; non-numeric labels are kept."""
    return str(int(value)) if value and value.isdigit() else (value or "")


def run_key(found):
    return (found.get("sub"), number(found.get("ses")), found.get("task"),
            found.get("acq", ""), number(found.get("run")))


def git_files(repository, pattern):
    listing = subprocess.run(["git", "-C", str(repository), "ls-files", pattern],
                             capture_output=True, text=True, check=True)
    return listing.stdout.split()


def git_commit(repository):
    result = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip()[:10]


def has_rows(path):
    """True when the file is present and has at least one row under its header."""
    if not path.is_file():
        return False
    with open(path, errors="ignore") as handle:
        return sum(1 for _ in handle) > 1


def timeseries_runs(timeseries_root):
    """{run key: where its parcel time series is and how long it is}."""
    runs = {}
    for path in sorted(Path(timeseries_root).glob(datasets.TIMESERIES_GLOB)):
        if not path.is_file():
            continue                      # annexed file that was not retrieved
        subject = path.name.split("_")[0][4:]
        with h5py.File(path) as handle:
            keys = []
            handle.visit(lambda key: keys.append(key)
                         if isinstance(handle[key], h5py.Dataset) else None)
            for key in keys:
                found = entities(Path(key).name)
                found["sub"] = subject
                runs[run_key(found)] = dict(
                    h5=str(path.relative_to(timeseries_root)), h5_key=key,
                    n_vols=handle[key].shape[0], ses_label=found.get("ses", ""))
    return runs


def closest_events(bold_path, bold_entities, events_by_folder, bids_root):
    """The events file that applies to a BOLD file under BIDS inheritance."""
    best = None
    folder = folder_key(Path(bold_path).parent)
    while True:
        for candidate in events_by_folder.get(folder, []):
            wanted = {key: value for key, value in entities(Path(candidate).name).items()
                      if key not in IGNORED_FOR_EVENTS}
            if not all(number(bold_entities.get(key)) == number(value)
                       for key, value in wanted.items()):
                continue
            filled = has_rows(bids_root / candidate)
            if best is None or (filled, len(wanted)) > (best[2], best[1]):
                best = (candidate, len(wanted), filled)
        if (best and best[2]) or folder == "":
            return best[0] if best else ""
        folder = folder_key(Path(folder).parent)


def folder_key(folder):
    """Folder as a dictionary key; the dataset root is the empty string."""
    return "" if str(folder) == "." else str(folder)


def sidecar(bids_root, bold_path):
    path = bids_root / re.sub(r"\.nii(\.gz)?$", ".json", bold_path)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def bids_runs(bids_root):
    """{run key: BOLD file, events file, TR and acquisition time} from the BIDS tree."""
    events_by_folder = {}
    for path in git_files(bids_root, "*_events.tsv"):
        if "desc-" not in Path(path).name:       # derived annotations are not the raw events
            events_by_folder.setdefault(folder_key(Path(path).parent), []).append(path)
    runs = {}
    for path in git_files(bids_root, "*_bold.nii.gz"):
        found = entities(Path(path).name)
        if found.get("part") == "phase" or found.get("echo") not in (None, "1"):
            continue
        meta = sidecar(bids_root, path)
        runs[run_key(found)] = dict(
            bold=path, events=closest_events(path, found, events_by_folder, bids_root),
            tr=meta.get("RepetitionTime"), acq_time=meta.get("AcquisitionTime", ""),
            ses_label=found.get("ses", ""))
    return runs


def merge_unmatched(series, bids):
    """Pair runs that differ only by a missing session level or by the task label."""
    notes = {}
    for key in [key for key in series if key not in bids]:
        subject, session, task, acquisition, run = key
        no_session = (subject, "", task, acquisition, run)
        if no_session in bids and no_session not in series:
            bids[key] = bids.pop(no_session)
            continue
        same_place = (subject, session, acquisition, run)
        similar = [other for other in bids if other not in series
                   and (other[0], other[1], other[3], other[4]) == same_place
                   and other[2].lower().startswith(task.lower()[:3])]
        if len(similar) == 1:
            bids[key] = bids.pop(similar[0])
            notes[key] = (f"task label differs: time series '{task}', BIDS '{similar[0][2]}'",
                          similar[0][2])
    return notes


def status_of(row):
    if row.get("h5") and row.get("bold"):
        return "ok" if row.get("events") else "no events file"
    return "bold without time series" if row.get("bold") else "time series without bold"


def build_row(dataset, key, series, bids, note, bids_root):
    subject, session, task, acquisition, run = key
    row = {**bids.get(key, {}), **series.get(key, {})}
    session_label = row.get("ses_label") or session
    task_label = note[1] if note else task
    events = row.get("events", "")
    present = bool(events) and (bids_root / events).is_file()
    text = note[0] if note else ""
    if present and not has_rows(bids_root / events):
        text = text or "events file is an empty template (header only)"
    run_label = f" run-{run}" if run else ""
    session_text = f"ses-{session_label}" if session_label else ""
    return dict(
        dataset=dataset, sub=f"sub-{subject}", ses=session_text, task=task_label,
        acq=acquisition, run=run, id=f"sub-{subject} {session_text} {task_label}{run_label}",
        n_vols=row.get("n_vols"), tr=row.get("tr") or settings.DEFAULT_TR,
        acq_time=row.get("acq_time", ""), h5=row.get("h5", ""), h5_key=row.get("h5_key", ""),
        bold=row.get("bold", ""), events=events, events_present=present,
        status=status_of(row), note=text)


def index_dataset(checkout, dataset):
    """The run index of one dataset as a DataFrame (see COLUMNS)."""
    bids_root = datasets.bids_dir(checkout, dataset)
    timeseries_root = datasets.timeseries_dir(checkout, dataset)
    series, bids = timeseries_runs(timeseries_root), bids_runs(bids_root)
    notes = merge_unmatched(series, bids)
    rows = [build_row(dataset, key, series, bids, notes.get(key), bids_root)
            for key in sorted(set(series) | set(bids), key=lambda k: tuple(str(x) for x in k))]
    table = pd.DataFrame(rows, columns=COLUMNS[:-2])
    table["bids_commit"] = git_commit(bids_root)
    table["timeseries_commit"] = git_commit(timeseries_root)
    return table


def write_index(checkout, dataset, output_path):
    table = index_dataset(checkout, dataset)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, sep="\t", index=False)
    return table


def read_index(path):
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    table["n_vols"] = pd.to_numeric(table.n_vols, errors="coerce")
    table["tr"] = pd.to_numeric(table.tr, errors="coerce").fillna(settings.DEFAULT_TR)
    table["events_present"] = table.events_present == "True"
    return table
