"""Problems that are already documented in public, to tell them from new findings.

`known_issues.tsv` (next to this file) lists runs with a public reference: a
GitHub issue of the dataset, or a notes file distributed with it. It only
annotates the results: a documented run is checked and flagged like any other,
and the dashboard marks its flag as documented.

Columns: dataset, sub, ses, runs ("all" or a comma-separated list of run
numbers), check (the check the note is about), reference, url, note.
"""

from pathlib import Path

import pandas as pd

TABLE = Path(__file__).with_name("known_issues.tsv")


def session_number(label):
    """'ses-002' and 'ses-02' are the same session."""
    digits = str(label).replace("ses-", "")
    return str(int(digits)) if digits.isdigit() else digits


def load(path=TABLE):
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    table["ses_number"] = table.ses.map(session_number)
    return table


def matches(issue, run):
    if issue.sub != run["sub"] or issue.ses_number != session_number(run["ses"]):
        return False
    return issue.runs == "all" or str(run["run"]) in issue.runs.split(",")


def annotate(runs, dataset, path=TABLE):
    """Add the columns known, known_check and known_url to a runs table."""
    issues = load(path)
    issues = issues[issues.dataset == dataset]
    notes, checks, urls = [], [], []
    for _, run in runs.iterrows():
        found = [issue for issue in issues.itertuples() if matches(issue, run)]
        notes.append("; ".join(f"{issue.note} ({issue.reference})" for issue in found))
        checks.append(",".join(sorted({issue.check for issue in found})))
        urls.append(found[0].url if found else "")
    runs = runs.copy()
    runs["known"], runs["known_check"], runs["known_url"] = notes, checks, urls
    return runs
