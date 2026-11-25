"""The steps of the alignment QA, one function each, from a cneuromod.all
checkout to the dashboard. Each step reads only what earlier steps wrote (or
the checkout itself), writes one output per dataset, and skips a dataset whose
output already exists.

    index        run index: time series, BOLD and events of each run
    scan         brain-signal scan (the slow step)
    repetitions  videogame datasets only: is each play's recording the right one?
    flags        findings per run, all checks
    tables       TSV tables of the findings, all datasets
    dashboard    one self-contained HTML page

`python -m analysis --help` runs them from the command line; `tasks.py` wraps
them as invoke tasks.
"""

from pathlib import Path

from analysis import bids_index, datasets, flags, scan

STEP_FOLDERS = {"index": "index", "scan": "scan", "repetitions": "repetitions",
                "flags": "flags", "tables": "tables", "dashboard": "dashboard"}


def output_path(output_dir, step, dataset):
    extension = "_runs.tsv" if step == "index" else ".pkl"
    return Path(output_dir) / STEP_FOLDERS[step] / f"{dataset}{extension}"


def selected(checkout, names=None):
    """The datasets to process: those asked for, or every installed one."""
    installed = datasets.available(checkout)
    if not names:
        return installed
    missing = [name for name in names if name not in installed]
    if missing:
        raise SystemExit(f"Not installed in {checkout}: {', '.join(missing)}. "
                         "Run `invoke fetch` first.")
    return list(names)


def require(path, step):
    if not Path(path).exists():
        raise SystemExit(f"{path} is missing: run the '{step}' step first.")


def run_index(checkout, output_dir, names=None):
    for dataset in selected(checkout, names):
        target = output_path(output_dir, "index", dataset)
        if target.exists():
            print(f"index: {dataset} skipped (output exists)")
            continue
        table = bids_index.write_index(checkout, dataset, target)
        print(f"index: {dataset} {len(table)} runs {table.status.value_counts().to_dict()}")


def run_scan(checkout, output_dir, names=None, subjects=None, n_jobs=6):
    for dataset in selected(checkout, names):
        target = output_path(output_dir, "scan", dataset)
        if target.exists():
            print(f"scan: {dataset} skipped (output exists)")
            continue
        index_path = output_path(output_dir, "index", dataset)
        require(index_path, "index")
        results = scan.scan_dataset(checkout, dataset, bids_index.read_index(index_path),
                                    subjects=subjects, n_jobs=n_jobs)
        scan.write_scan(results, target)
        print(f"scan: {dataset} {len(results['index'])} runs, "
              f"{len(results['runs'])} predictor fits")


def run_repetitions(checkout, output_dir, names=None, n_jobs=6):
    from analysis.checks import repetitions

    for dataset in selected(checkout, names):
        if datasets.profile(dataset).kind != "videogame":
            continue
        target = output_path(output_dir, "repetitions", dataset)
        if target.exists():
            print(f"repetitions: {dataset} skipped (output exists)")
            continue
        scan_path = output_path(output_dir, "scan", dataset)
        require(scan_path, "scan")
        results = repetitions.check_dataset(checkout, dataset, scan.read_scan(scan_path),
                                            n_jobs=n_jobs)
        repetitions.write(results, target)
        print(f"repetitions: {dataset} {len(results['plays'])} plays")


def run_flags(output_dir, names):
    for dataset in names:
        target = output_path(output_dir, "flags", dataset)
        if target.exists():
            print(f"flags: {dataset} skipped (output exists)")
            continue
        scan_path = output_path(output_dir, "scan", dataset)
        index_path = output_path(output_dir, "index", dataset)
        require(scan_path, "scan")
        findings = flags.flag_dataset(scan.read_scan(scan_path),
                                      bids_index.read_index(index_path))
        flags.write_flags(findings, target)
        print(f"flags: {dataset} {len(findings['runs'])} runs")


def scanned(output_dir):
    """Datasets that have a scan output, in name order."""
    folder = Path(output_dir) / STEP_FOLDERS["scan"]
    return sorted(path.stem for path in folder.glob("*.pkl"))


def run_tables(output_dir):
    from analysis import export

    names = [name for name in scanned(output_dir)
             if output_path(output_dir, "flags", name).exists()]
    export.write_tables(output_dir, names)


def run_dashboard(output_dir):
    from analysis import export
    from analysis.dashboard import build

    names = [name for name in scanned(output_dir)
             if output_path(output_dir, "flags", name).exists()]
    data = export.dashboard_data(output_dir, names)
    target = Path(output_dir) / STEP_FOLDERS["dashboard"] / "qa_align.html"
    build.write_page(data, target)
    print(f"dashboard: {target} ({target.stat().st_size / 1e6:.1f} MB)")
