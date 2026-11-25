"""Retrieve what the checks read from a cneuromod.all checkout, and nothing else.

For each dataset: install its `bids` and `timeseries` subdatasets (never
recursively: their own subdatasets stay uninstalled), then get

  timeseries   the parcel time series, one HDF5 file per subject
  bids         the events files; for videogame datasets also the replay of
               each play (bk2) and its frame-wise features (*_lowlevel.npy)

BOLD images are never downloaded: the sidecars the checks read are plain git
files. Content on a remote the user cannot reach is skipped with a warning and
remembered, so a repeated fetch does not wait for it again.

A subdataset is installed at the commit cneuromod.all pins and is left there:
the version that is checked is the version cneuromod.all distributes. To check
newer data, update the cneuromod.all checkout itself.
"""

from pathlib import Path

from airoh.datalad import (
    datalad_get,
    load_known_failures,
    prefetch_pattern,
    save_known_failures,
)

from analysis import datasets

TIMESERIES_PATTERN = "*atlas-cneuromod26*_timeseries.h5"
TIMESERIES_FOLDER = "timeseries/cneuromod2026"
BIDS_PATTERNS = ["*_events.tsv"]
VIDEOGAME_PATTERNS = ["*.bk2", "*_lowlevel.npy"]


def outside_git(path):
    """Annexed content also lives under .git/annex: only match the working tree."""
    return ".git" not in path.parts


def fetch_patterns(root, patterns, cache_dir, subdir=""):
    """Get every file matching the patterns under one subdataset. Returns counts.

    Paths that could not be retrieved are remembered in `cache_dir` (outside the
    subdataset, which must stay clean), keyed by the subdataset they belong to.
    """
    prefix = f"{Path(root).parent.name}/{Path(root).name}/"
    all_failures = load_known_failures(cache_dir)
    known = {path[len(prefix):] for path in all_failures if path.startswith(prefix)}
    totals = dict(present=0, fetched=0, skipped=0, failed=0)
    for pattern in patterns:
        present, fetched, skipped, failed, resolved = prefetch_pattern(
            root, pattern, subdir=subdir, skip_set=known, match=outside_git)
        known = (known - resolved) | failed
        totals["present"] += present
        totals["fetched"] += fetched
        totals["skipped"] += skipped
        totals["failed"] += len(failed)
    elsewhere = {path for path in all_failures if not path.startswith(prefix)}
    save_known_failures(cache_dir, elsewhere | {prefix + path for path in known})
    return totals


def install_pinned(checkout, path, strict=False):
    """Install one subdataset (tree only) at its pinned commit; leave it if already there."""
    if datasets.is_installed(checkout / path):
        return
    datalad_get(path, checkout, get_content=False, strict=strict)
    if strict and not datasets.is_installed(checkout / path):
        raise RuntimeError(f"subdataset {path} could not be installed in {checkout}")


def fetch_dataset(checkout, dataset, cache_dir, strict=False):
    """Install and fill the two subdatasets of one dataset. Returns a summary dict."""
    checkout = Path(checkout)
    for name in (datasets.BIDS_SUBDATASET, datasets.TIMESERIES_SUBDATASET):
        install_pinned(checkout, f"{dataset}/{name}", strict=strict)
    summary = {}
    timeseries_root = datasets.timeseries_dir(checkout, dataset)
    if datasets.is_installed(timeseries_root):
        summary["timeseries"] = fetch_patterns(timeseries_root, [TIMESERIES_PATTERN],
                                               cache_dir, subdir=TIMESERIES_FOLDER)
    bids_root = datasets.bids_dir(checkout, dataset)
    if datasets.is_installed(bids_root):
        patterns = list(BIDS_PATTERNS)
        if datasets.profile(dataset).kind == "videogame":
            patterns += VIDEOGAME_PATTERNS
        summary["bids"] = fetch_patterns(bids_root, patterns, cache_dir)
    if strict:
        require_complete(dataset, summary)
    return summary


def require_complete(dataset, summary):
    """Raise unless both subdatasets are installed and every file was retrieved."""
    for part in ("timeseries", "bids"):
        counts = summary.get(part)
        if counts is None:
            raise RuntimeError(f"{dataset}/{part} is not installed")
        missing = counts["failed"] + counts["skipped"]
        if missing or counts["present"] + counts["fetched"] == 0:
            raise RuntimeError(
                f"{dataset}/{part}: {missing} files could not be retrieved "
                f"({counts['present'] + counts['fetched']} present). "
                "See source_data/CONTENT.md, 'Access'.")


def fetch_all(checkout, cache_dir, names=None, strict=False):
    """Fetch the datasets given, or every dataset cneuromod.all registers with a time series.

    `cache_dir` is where the list of unreachable files is kept between fetches
    (`.fetch_failures.json`); delete that file to try them again.
    """
    for dataset in names or datasets.declared(checkout):
        summary = fetch_dataset(checkout, dataset, cache_dir, strict=strict)
        parts = [f"{part}: {counts['present'] + counts['fetched']} files"
                 + (f", {counts['failed'] + counts['skipped']} not reachable"
                    if counts["failed"] + counts["skipped"] else "")
                 for part, counts in summary.items()]
        print(f"fetch: {dataset} " + ("; ".join(parts) or "subdatasets not installed"))
