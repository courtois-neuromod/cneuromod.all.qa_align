from pathlib import Path

from invoke import task

SMOKE_DATASET = "floc"     # the smallest dataset: 36 short runs, a few megabytes


def checkout_dir(c):
    """Where the cneuromod.all checkout lives (see `datasets:` in invoke.yaml)."""
    entry = c.config.get("datasets")["cneuromod"]
    return Path(entry if isinstance(entry, str) else entry["output_dir"])


def output_dir(c):
    return Path(c.config.get("output_data_dir"))


def dataset_list(datasets):
    """'floc,things' -> ['floc', 'things']; None stays None (meaning every dataset)."""
    return [name.strip() for name in datasets.split(",") if name.strip()] if datasets else None


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
@task(help={
    "source": "Path to an existing cneuromod.all checkout to symlink instead of cloning.",
    "datasets": "Comma-separated datasets to retrieve (default: every dataset with time series).",
    "strict": "Fail on the first file that cannot be retrieved instead of skipping it.",
})
def fetch_cneuromod(c, source=None, datasets=None, strict=False):
    """
    Make cneuromod.all available and retrieve the files the checks read.

    Clones cneuromod.all (or symlinks an existing checkout given with
    --source), installs the `bids` and `timeseries` subdatasets of each
    dataset — never recursively — and gets the parcel time series, the events
    files and, for videogame datasets, the replays and their frame-wise
    features. BOLD images are not downloaded.

    Tolerant by default: a file on a remote you cannot reach is skipped with a
    warning and listed in source_data/.fetch_failures.json, so that the next
    fetch does not wait for it again (delete that file to retry). See
    source_data/CONTENT.md for the access the data needs.
    """
    from airoh.datalad import install_dataset

    from analysis import fetching

    install_dataset(c, "cneuromod", source=source)
    fetching.fetch_all(checkout_dir(c), Path(c.config.get("source_data_dir")),
                       dataset_list(datasets), strict=strict)

@task(help={
    "cneuromod_source": "Path to an existing cneuromod.all checkout to symlink instead of cloning.",
    "datasets": "Comma-separated datasets to retrieve (default: every dataset with time series).",
})
def fetch(c, cneuromod_source=None, datasets=None):
    """
    Retrieve all data assets. Each asset has its own fetch-{name} task; this
    umbrella task routes a per-asset --{name}-source flag to the matching one.

    Records what each asset actually resolved to in source_data/MANIFEST.json,
    so the inputs a later run consumed stay identifiable — including the commit
    of a symlinked external checkout. See CLAUDE.md, "Recording asset versions".
    """
    from airoh.provenance import record_sources

    fetch_cneuromod(c, source=cneuromod_source, datasets=datasets)
    record_sources(c)

# --------------------------------------------------------------------------- #
# Analysis steps
# --------------------------------------------------------------------------- #
@task(help={"datasets": "Comma-separated datasets (default: every installed dataset)."})
def run_index(c, datasets=None):
    """
    Match every run of the parcel time series with its BOLD and events files.

    One table per dataset in output_data/index/. Skipped for a dataset whose
    table already exists — every run step caches that way; use
    `invoke clean-index` (or `invoke run --force`) to redo it.
    """
    from analysis import pipeline

    pipeline.run_index(checkout_dir(c), output_dir(c), dataset_list(datasets))

@task(help={
    "datasets": "Comma-separated datasets (default: every installed dataset).",
    "subjects": "Comma-separated subjects, e.g. sub-01,sub-02 (default: all).",
    "jobs": "Worker processes, one subject each (default: 6).",
    "smoke": "Scan only the smallest dataset.",
})
def run_scan(c, datasets=None, subjects=None, jobs=6, smoke=False):
    """
    Brain-signal scan: fit of every predictor with the BOLD at every shift.

    The slow step (about 15 minutes for every dataset on 6 cores). One file per
    dataset in output_data/scan/; a dataset whose file exists is skipped.
    Restricting --subjects is for trying things out: the result then holds
    those subjects only, and the "other subjects" predictor needs at least 3.
    """
    from analysis import pipeline

    names = [SMOKE_DATASET] if smoke else dataset_list(datasets)
    pipeline.run_scan(checkout_dir(c), output_dir(c), names,
                      subjects=dataset_list(subjects), n_jobs=int(jobs))

@task(help={
    "datasets": "Comma-separated videogame datasets (default: every installed one).",
    "jobs": "Worker processes, one subject each (default: 6).",
})
def run_repetitions(c, datasets=None, jobs=6):
    """
    Videogame datasets: check that each play points at its own replay.

    One file per videogame dataset in output_data/repetitions/; other datasets
    have no replay files and are left out. Needs the scan of the dataset.
    """
    from analysis import pipeline

    pipeline.run_repetitions(checkout_dir(c), output_dir(c), dataset_list(datasets),
                             n_jobs=int(jobs))

@task(help={"datasets": "Comma-separated datasets (default: every scanned dataset)."})
def run_flags(c, datasets=None):
    """
    Turn each dataset's scan into findings: start shift, pairing, end of run, drift.

    One file per dataset in output_data/flags/. Reads the scan only, so it is
    quick: after changing a threshold in analysis/settings.py, run
    `invoke clean-flags` then `invoke run`.
    """
    from analysis import pipeline

    names = dataset_list(datasets) or pipeline.scanned(output_dir(c))
    pipeline.run_flags(output_dir(c), names)

@task
def run_tables(c):
    """
    Write the findings of every dataset as TSV tables in output_data/tables/.

    Always rewritten: it is quick, and the tables must match the flags.
    """
    from analysis import pipeline

    pipeline.run_tables(output_dir(c))

@task
def run_dashboard(c):
    """
    Build the dashboard: output_data/dashboard/qa_align.html.

    One self-contained page (data inside, no network needed). Always rebuilt:
    it is quick, and the page must match the flags.
    """
    from analysis import pipeline

    pipeline.run_dashboard(output_dir(c))

# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@task(help={
    "force": "Delete every computed output first, then run from scratch.",
    "datasets": "Comma-separated datasets (default: every installed dataset).",
    "jobs": "Worker processes for the scan and the repetitions check (default: 6).",
})
def run(c, force=False, datasets=None, jobs=6):
    """
    Full pipeline: index → scan → repetitions → flags → tables → dashboard.

    Steps are called directly rather than through `pre=`, so that flags like
    --force reach them: a `pre=` chain runs before this body, which would be
    too late.

    The per-dataset steps cache by checking whether their output already
    exists, so a repeated `run` only rebuilds the tables and the dashboard.
    That is deliberate — but it also means edited code will NOT re-run on its
    own. `--force` is the sledgehammer: clean everything, then start over. To
    redo one step, call its `clean-{name}` task and run again.
    """
    from airoh.provenance import record_run

    if force:
        print("💥 --force: removing every computed output before running")
        clean(c)
    run_index(c, datasets=datasets)
    run_scan(c, datasets=datasets, jobs=jobs)
    run_repetitions(c, datasets=datasets, jobs=jobs)
    run_flags(c, datasets=datasets)
    run_tables(c)
    run_dashboard(c)
    record_run(c, tasks="run-index,run-scan,run-repetitions,run-flags,run-tables,run-dashboard")
    print("all analyses completed")

@task
def run_smoke(c):
    """
    Smoke test: a minimal end-to-end pass over the whole pipeline.

    Fetches and processes the smallest dataset only, into
    output_data/smoke/, so it never mixes with real results. Retrieval is
    strict here: if the data cannot be fetched, the test must fail rather than
    pass on an empty result.
    """
    from analysis import pipeline

    fetch_cneuromod(c, datasets=SMOKE_DATASET, strict=True)
    smoke_dir = output_dir(c) / "smoke"
    checkout = checkout_dir(c)
    pipeline.run_index(checkout, smoke_dir, [SMOKE_DATASET])
    pipeline.run_scan(checkout, smoke_dir, [SMOKE_DATASET], n_jobs=2)
    pipeline.run_repetitions(checkout, smoke_dir, [SMOKE_DATASET], n_jobs=2)
    pipeline.run_flags(smoke_dir, [SMOKE_DATASET])
    pipeline.run_tables(smoke_dir)
    pipeline.run_dashboard(smoke_dir)
    page = smoke_dir / "dashboard" / "qa_align.html"
    if not page.is_file():
        raise RuntimeError(f"Smoke test failed: {page} was not written.")
    print("✅ Smoke test complete.")

@task(help={
    "skip": "Comma-separated check names to skip.",
    "strict": "Treat warnings as failures.",
})
def verify(c, skip=None, strict=False):
    """
    Check that the code, config, data and docs still agree.

    Run this before committing. It is deliberately NOT part of `run`:
    reproducing results should never depend on documentation hygiene. See
    CLAUDE.md, "Verification", for what each check covers.
    """
    from airoh.verify import verify as airoh_verify
    airoh_verify(c, skip=skip, strict=strict)

# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
def remove_step(c, step):
    """Remove the output folder of one step (and the smoke test's copy)."""
    import shutil

    for folder in (output_dir(c) / step, output_dir(c) / "smoke" / step):
        if folder.exists():
            shutil.rmtree(folder)
            print(f"🧹 Removed: {folder}")

@task
def clean_index(c):
    """
    Remove the run indexes (output_data/index/).
    """
    remove_step(c, "index")

@task
def clean_scan(c):
    """
    Remove the scans (output_data/scan/). Recomputing them is the slow step.
    """
    remove_step(c, "scan")

@task
def clean_repetitions(c):
    """
    Remove the repetitions measures (output_data/repetitions/).
    """
    remove_step(c, "repetitions")

@task
def clean_flags(c):
    """
    Remove the findings (output_data/flags/).
    """
    remove_step(c, "flags")

@task
def clean_tables(c):
    """
    Remove the TSV tables (output_data/tables/).
    """
    remove_step(c, "tables")

@task
def clean_dashboard(c):
    """
    Remove the dashboard (output_data/dashboard/).
    """
    remove_step(c, "dashboard")

@task
def clean(c):
    """
    Remove all computed outputs.

    The steps are called in the body rather than declared as `pre=`, because a
    `pre=` chain only fires when invoke runs the task from the command line.
    Calling `clean(c)` from Python — which is what `run --force` does — would
    otherwise execute an empty function and silently delete nothing.
    """
    clean_index(c)
    clean_scan(c)
    clean_repetitions(c)
    clean_flags(c)
    clean_tables(c)
    clean_dashboard(c)

@task
def clean_cneuromod(c):
    """
    Remove the cneuromod.all source asset (the checkout, or the symlink to one).

    Not called by `clean` or `run --force` — those only touch output_data/.
    A symlink is unlinked. A real checkout is left in place and reported:
    its git-annex content is write-protected, and deleting gigabytes of data
    should be your decision, not a side effect.
    """
    checkout = checkout_dir(c)
    if checkout.is_symlink():
        checkout.unlink()
        print(f"🧹 Removed link: {checkout}")
    elif checkout.exists():
        print(f"🫧 {checkout} is a real checkout, not a link: left in place. "
              "Remove it yourself if you mean to (git-annex content is write-protected).")
    else:
        print(f"🫧 Skipping: {checkout} does not exist.")

@task
def clean_source(c):
    """
    Remove all source data assets. Body calls each clean-{name} task.
    """
    clean_cneuromod(c)
