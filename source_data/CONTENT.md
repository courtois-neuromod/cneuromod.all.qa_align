# 📁 Source Data Contents

After `invoke fetch` is complete, expect the following content:

- `cneuromod.all/` — a checkout of
  [cneuromod.all](https://github.com/courtois-neuromod/cneuromod.all), the
  DataLad superdataset of the Courtois NeuroMod project, or a symlink to a
  checkout you already have (`invoke fetch --cneuromod-source /path`). For every
  dataset that has parcel time series, two of its subdatasets are installed —
  and only those two, never recursively:
  - `<dataset>/timeseries/` — parcel time series. The checks read
    `timeseries/cneuromod2026/sub-*/*atlas-cneuromod26*_timeseries.h5`, one
    HDF5 file per subject holding one array (volumes × parcels) per run.
  - `<dataset>/bids/` — the raw BIDS dataset. The checks read the events files
    (`*_events.tsv`), the BOLD sidecars (`*_bold.json`: repetition time and
    acquisition time; plain git files, nothing to download) and the list of
    BOLD files. For the videogame datasets (`mario`, `mariostars`, `mario3`,
    `shinobi`) they also read, for each play, its replay (`gamelogs/*.bk2`) and
    its frame-wise features (`gamelogs/*_lowlevel.npy`).
- `MANIFEST.json` — what the asset actually resolved to: the URL or the real
  path behind a symlink, and the commit of the checkout. Written by
  `invoke fetch`. The commit of each `bids` and `timeseries` subdataset is
  recorded per run in `output_data/index/`.

BOLD images, anatomical images and the other derivatives are never downloaded.
The files the checks read amount to about 11 GB for every dataset together:
9 GB of time series, the rest replays and their features.

## Access

🔐 The git repositories are public, but the file content is not: it lives on
the CNeuroMod S3 store and needs credentials. Access is granted on request —
see the access page of the CNeuroMod documentation, <https://docs.cneuromod.ca>.
Once you have a key pair:

```bash
export AWS_ACCESS_KEY_ID=<your key>  AWS_SECRET_ACCESS_KEY=<your secret>
invoke fetch
```

Without credentials, `invoke fetch` still clones cneuromod.all and installs the
subdatasets, gets the few files that are publicly reachable (some events
files), then warns for the rest and carries on. The time series are not public,
so `invoke run` then finds nothing to check. An empty result of that kind means
missing access, not a broken pipeline. The files that could not be reached are
listed in `.fetch_failures.json` and skipped by later fetches: delete that file
once you have access.

Two datasets (`narratives`, `ood`) keep their events files on a store that is
not reachable from outside the lab. Their runs are then checked with the
"other subjects" predictor only, which needs no events file.

📝 Note: this folder is **ignored by Git** except for `MANIFEST.json` and this
file. The checkout is a DataLad dataset with its own history; nothing under it
is ever committed here. The data must not be redistributed: it is covered by
the CNeuroMod data use agreement.

📝 Note: if `cneuromod.all` is a symlink, the manifest records what it pointed
at, so a linked input stays identifiable. A symlinked checkout is used as it is:
`invoke fetch` installs and gets the missing pieces inside it.
