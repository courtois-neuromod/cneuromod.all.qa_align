# 📁 Output Data Contents

Once the pipeline is run, this folder will contain the following. `<dataset>`
is the name of a cneuromod.all dataset (`friends`, `mario`, `things`, ...).

- `index/<dataset>_runs.tsv` — one row per run found in the parcel time series
  or in the BIDS dataset: its time series (file and key), BOLD file, events
  file, number of volumes, repetition time, acquisition time, a coverage
  `status` saying what is missing, and the commits of the `bids` and
  `timeseries` subdatasets it was read from. Written by `invoke run-index`.
- `scan/<dataset>.pkl` — the brain-signal scan: for every run and predictor, the
  fit at each shift, the fit with the events of nearby runs, and the best shift
  in sliding windows (also with simulated jumps and drifts). A Python pickle of
  plain dicts, lists and numpy arrays; read it with `analysis.scan.read_scan`.
  Written by `invoke run-scan`, the slow step.
- `repetitions/<dataset>.pkl` — videogame datasets only: for every play, how
  well its replay and the other replays of the same level fill its slot and
  fit the brain. Read it with `analysis.checks.repetitions.read`. Written by
  `invoke run-repetitions`.
- `flags/<dataset>.pkl` — the findings: one row per run with the result of the
  start-shift, pairing, end-of-run and drift checks, which predictors are
  reliable for each task, and what the simulations say each check can see.
  Read it with `analysis.flags.read_flags`. Written by `invoke run-flags`.
- `tables/` — the findings of every dataset as TSV, written by
  `invoke run-tables`:
  - `overview.tsv` — one row per dataset: runs, predictors that count, and per
    check how many runs were judged and how many were flagged.
  - `flagged_runs.tsv` — one row per flagged run and check, in plain words,
    with `status` = `documented` (a public reference exists, see
    `analysis/known_issues.tsv`) or `new`.
  - `runs.tsv` — one row per run with the result of every check and the shift,
    fit and slope given by each predictor.
  - `predictors.tsv` — one row per dataset, task and predictor: usual shift,
    agreement between runs, reliable or not.
  - `repetitions.tsv` — one row per flagged play of the videogame datasets.
  - `coverage.tsv` — the runs that could not be checked, and why.
- `dashboard/qa_align.html` — the dashboard: one self-contained page with its
  data inside, to open in a browser (no network needed). Written by
  `invoke run-dashboard`.
- `smoke/` — the same folders for the smallest dataset only, written by
  `invoke run-smoke`.
- `PROVENANCE.json` — what produced everything above: the project's git commit,
  the environment, the input manifest it consumed, and a checksum of every
  output file. Written by `invoke run`.

📝 Note: everything here except `PROVENANCE.json` and this file is **ignored by
Git** (see `.gitignore`): the results are rebuilt by `invoke run`, and the
tables and the dashboard name individual runs of a dataset under a data use
agreement, so sharing them is a decision to take outside this repository.
`PROVENANCE.json` changes on every run, by design.

📝 Note: the per-dataset files (`index/`, `scan/`, `repetitions/`, `flags/`) are
the cache: a dataset whose file exists is skipped. `tables/` and `dashboard/`
are rewritten on every `invoke run`.
