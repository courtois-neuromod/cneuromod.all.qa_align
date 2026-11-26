# cneuromod.all.qa_align

Quality assessment of the alignment between the data streams of the
[Courtois NeuroMod](https://www.cneuromod.ca/) datasets: does each BOLD run
start with its events, do the events belong to that run, do both end together,
and does the timing hold from the start of the run to its end?

The checks read the timing **from the brain signal itself**. The events of a
run are turned into a predicted BOLD signal, which is slid along the measured
one; the shift at which they fit best should be the same in every run of a
task. No task log is needed, so the same checks apply to every dataset of
[cneuromod.all](https://github.com/courtois-neuromod/cneuromod.all), from a
clean clone to one dashboard:

```bash
uv sync
uv run invoke fetch
uv run invoke run
# then open output_data/dashboard/qa_align.html
```

This repository follows the [`airoh`](https://pypi.org/project/airoh/)
template: an [`invoke`](https://www.pyinvoke.org/) task runner around plain
Python modules in `analysis/`.

---

## 🔍 What is checked

| Check | Question | Evidence |
|---|---|---|
| **Start shift** | Did the BOLD start with the events' t=0? | Best shift of each predictor against the usual one of its task; the scanner's and the task's clocks where the events keep the task clock |
| **Pairing** | Are these the events of this run? | Fit with the run's own events against the events of the neighbouring runs |
| **End of run** | Do events and BOLD end together? | Time between the last event and the end of the BOLD |
| **Drift** | Does the timing hold along the run? | Best shift in sliding windows |
| **Repetitions** | Does each play point at its own replay? | Videogame datasets: replay length against its slot in the events, and fit with the brain |
| **Coverage** | What could not be checked? | Runs without time series, BOLD or events |

A check uses several **predictors**, each a different reading of the same run:

- from the events of any dataset: *stimulus on/off*, the main *conditions*, and
  *button presses*;
- where every subject received the same stimulus: *other subjects* — the mean
  BOLD of the other subjects, which needs no events file and so tells an events
  problem from a BOLD problem;
- for videogame datasets, from the replay of each play: *button presses*,
  *screen luminance*, *optical flow* and *audio envelope*.

Not every predictor works everywhere, so each one only votes for a task where
it is **reliable** (most runs agree on the same shift), and a run's result
always comes from all its reliable predictors together: a real shift moves
(nearly) all of them. Each check is also run on simulated faults, to state what
it can and cannot see in each dataset.

Not every check is possible for every dataset. The dashboard says which ones
ran, and why the others could not:

| Dataset family | Predictors | Checks that do not apply |
|---|---|---|
| Videogames (`mario`, `mariostars`, `mario3`, `shinobi`) | game on/off, button presses, luminance, optical flow, audio; clocks where the events keep the task clock | — |
| Naturalistic (`friends`, `movie10`, `narratives`, `ood`, `harrypotter`, `petit-prince`) | other subjects; stimulus on/off where the events are available | Repetitions |
| Tasks (`floc`, `hcptrt`, `things`, `triplets`, `retinotopy`, `langlocalizer`, `gamepad`, ...) | stimulus on/off, conditions, button presses, other subjects | Repetitions |
| Different stimulus per subject and no usable events (`multfs`, `mutemusic`) | none | Everything but Coverage |

A dataset that is not known to `analysis/datasets.py` is checked with the
defaults, so a new dataset of cneuromod.all works without changing the code.

---

## 🚀 Quick Start

### **Step 1**: Install dependencies

Using `uv` (recommended):
```bash
uv sync
```
This creates a `.venv` and installs all dependencies from `pyproject.toml`.

Using `pip` (e.g. in a virtual environment):
```bash
pip install -r requirements.txt
```

Using `conda`:
```bash
conda env create -n airoh_env -f environment.yml
conda activate airoh_env
```

The data is retrieved with [DataLad](https://www.datalad.org/) and git-annex,
which the dependencies install.

---

### **Step 2**: Fetch the source data

```bash
export AWS_ACCESS_KEY_ID=<your key>  AWS_SECRET_ACCESS_KEY=<your secret>
invoke fetch
```

Clones cneuromod.all into `source_data/`, installs the `bids` and `timeseries`
subdatasets of every dataset (never recursively), and gets the files the checks
read: parcel time series, events files and, for the videogame datasets, the
replays. About 11 GB; no BOLD image is downloaded. The data needs CNeuroMod
credentials — see `source_data/CONTENT.md` for how to get them and for what a
fetch without access looks like.

Already have a checkout of cneuromod.all? Link it instead of cloning:

```bash
invoke fetch --cneuromod-source /path/to/cneuromod.all
```

To retrieve some datasets only: `invoke fetch --datasets floc,things`.

---

### **Step 3**: Run the full pipeline

```bash
invoke run
```

Runs every step in order, for every dataset that was fetched: index → scan →
repetitions → flags → tables → dashboard. The scan is the slow step: about
20 minutes for all datasets on 6 cores (`--jobs` changes that). `--datasets
floc,things` restricts the run.

Steps that have already produced their output for a dataset are skipped — only
missing outputs are recomputed. That caching is by file existence, not by
content: **a step you just edited will still be skipped**, because its old
output is sitting right there. To rebuild everything:

```bash
invoke run --force    # clean everything, then run from scratch
```

To redo a single step, remove its outputs and run again. The usual case is a
changed threshold in `analysis/settings.py`, which only needs the quick steps:

```bash
invoke clean-flags
invoke run
```

`invoke run` also writes `output_data/PROVENANCE.json`, recording the project's
git commit, the environment, the inputs it consumed and a checksum of every
output — so a result stays traceable to whatever produced it.

---

### **Step 4**: Read the results

Open `output_data/dashboard/qa_align.html` in a browser. It is one
self-contained page: the Overview lists every dataset against every check and
every flagged run; each check has its own tab with its method, its results and
what a normal run looks like. The same findings are in `output_data/tables/` as
TSV. See `output_data/CONTENT.md` for every file.

A flagged run is a run to look at, not a verdict. Runs whose problem already
has a public reference (a GitHub issue of the dataset, a notes file distributed
with it) are marked as documented: they are listed in
`analysis/known_issues.tsv`, which you extend by adding a row.

---

### **Step 5**: Check that everything still agrees

```bash
invoke verify
```

Compares the project against its own documentation: the task list in this
README, the packages in `requirements.txt` versus `pyproject.toml`, the paths
the docs mention, each data folder against its `CONTENT.md`, config keys, the
size and type of what git tracks, and the linter. It exits non-zero if anything
has drifted.

Run it before committing. It is deliberately not part of `invoke run` —
reproducing results should never depend on the documentation being tidy.

The unit tests build a small synthetic dataset with a planted shift and a
planted swap, and need no data: `uv run --with pytest python -m pytest tests`.

---

### **Step 6**: Clean outputs

```bash
invoke clean          # remove all outputs
invoke clean-{name}   # remove outputs of one specific step
invoke clean-source   # unlink the cneuromod.all checkout (a real clone is left in place)
```

---

## 🧰 Task overview

| Task | Description |
|---|---|
| `fetch` | Retrieve all data assets and record them in `source_data/MANIFEST.json` |
| `fetch-cneuromod` | Clone or link cneuromod.all, install the subdatasets, get the files the checks read |
| `run` | Full pipeline: index → scan → repetitions → flags → tables → dashboard |
| `run-index` | Match every run of the parcel time series with its BOLD and events files |
| `run-scan` | Brain-signal scan: fit of every predictor with the BOLD at every shift (slow) |
| `run-repetitions` | Videogame datasets: check that each play points at its own replay |
| `run-flags` | Turn each dataset's scan into findings |
| `run-tables` | Write the findings of every dataset as TSV tables |
| `run-dashboard` | Build `output_data/dashboard/qa_align.html` |
| `run-smoke` | Fast end-to-end pass on the smallest dataset, into `output_data/smoke/` |
| `verify` | Check that the code, config, data and docs still agree |
| `clean` | Remove all computed outputs |
| `clean-index` | Remove the run indexes |
| `clean-scan` | Remove the scans |
| `clean-repetitions` | Remove the repetitions measures |
| `clean-flags` | Remove the findings |
| `clean-tables` | Remove the TSV tables |
| `clean-dashboard` | Remove the dashboard |
| `clean-cneuromod` | Unlink the cneuromod.all checkout |
| `clean-source` | Remove all source data assets |

Use `invoke --list` or `invoke --help <task>` for details.

---

## 🧠 Design principles

Airoh projects follow a few conventions that keep analyses fast, reproducible,
and easy to pick up:

- **Fetch retrieves, run reads.** `invoke run` never downloads anything: a run
  whose files are missing is reported in the Coverage tab.
- **Analysis in code.** Computation lives in `analysis/` Python modules and is
  run by `invoke` tasks; the dashboard only displays what the tables contain.
- **Idempotent steps.** Each per-dataset step checks whether its output already
  exists and skips if it does. The flip side: caching is by existence, so
  `invoke run --force` is how you rebuild after editing something.
- **Mirrored clean tasks.** Every `run-{name}` has a matching `clean-{name}`
  that removes only its outputs. The top-level `clean` calls them all.
- **Smoke test.** `invoke run-smoke` does a fast minimal pass to verify the
  pipeline end-to-end.

## 🗂️ Code map

| Module | Role |
|---|---|
| `analysis/settings.py` | Every grid, range and threshold, with its meaning |
| `analysis/datasets.py` | What differs between datasets: kind, comparable runs, shared stimuli |
| `analysis/fetching.py` | What to install and get from cneuromod.all |
| `analysis/bids_index.py` | Run index: time series ↔ BOLD ↔ events |
| `analysis/predictors/` | Predictors from events, from game replays, from the other subjects |
| `analysis/scan.py` | The brain-signal scan |
| `analysis/checks/` | One module per check |
| `analysis/flags.py`, `analysis/findings.py` | Findings per run, and their wording |
| `analysis/known_issues.tsv` | Runs with a public reference |
| `analysis/export.py`, `analysis/dashboard/` | Tables and the dashboard |
| `analysis/pipeline.py` | The steps, as plain functions |
