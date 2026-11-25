"""The whole pipeline on a synthetic dataset with two planted faults."""

import json

import pytest

from analysis import datasets, flags, pipeline
from analysis.checks import pairing, shift
from tests import toy_dataset


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    checkout = toy_dataset.build(root)
    output_dir = root / "output_data"
    pipeline.run_index(checkout, output_dir)
    pipeline.run_scan(checkout, output_dir, n_jobs=2)
    pipeline.run_repetitions(checkout, output_dir)
    pipeline.run_flags(output_dir, pipeline.scanned(output_dir))
    pipeline.run_tables(output_dir)
    return checkout, output_dir


def toy_runs(outputs):
    found = flags.read_flags(pipeline.output_path(outputs[1], "flags", "toy"))
    return found["runs"].set_index("id")


def run_id(subject, run):
    return f"sub-{subject} ses-001 toy run-{run}"


def test_checkout_is_discovered(outputs):
    checkout, _ = outputs
    assert datasets.available(checkout) == ["toy"]
    assert datasets.declared(checkout) == ["toy"]


def test_every_run_is_indexed_and_scanned(outputs):
    runs = toy_runs(outputs)
    assert len(runs) == len(toy_dataset.SUBJECTS) * len(toy_dataset.RUNS)
    assert (runs.n_vols == toy_dataset.N_VOLUMES).all()


def test_planted_shift_is_found_with_its_size(outputs):
    runs = toy_runs(outputs)
    shifted = runs.loc[run_id(*toy_dataset.SHIFTED)]
    assert shift.is_flagged(shifted.flag_shift)
    assert abs(shifted.shift_vols - toy_dataset.SHIFT_VOLUMES) <= 0.5
    assert "stim" in shifted.backers.split(",")


def test_planted_swap_is_found_with_its_partner(outputs):
    runs = toy_runs(outputs)
    subject, first, second = toy_dataset.SWAPPED
    assert runs.loc[run_id(subject, first)].flag_pair == pairing.FLAGGED
    assert runs.loc[run_id(subject, first)].pair_other == run_id(subject, second)
    assert runs.loc[run_id(subject, second)].pair_other == run_id(subject, first)


def test_clean_runs_are_not_flagged(outputs):
    runs = toy_runs(outputs)
    planted = {run_id(*toy_dataset.SHIFTED), run_id(toy_dataset.SWAPPED[0], toy_dataset.SWAPPED[1]),
               run_id(toy_dataset.SWAPPED[0], toy_dataset.SWAPPED[2])}
    clean = runs[~runs.index.isin(planted)]
    assert not clean.flag_shift.map(shift.is_flagged).any()
    assert (clean.flag_pair != pairing.FLAGGED).all()


def test_tables_list_the_planted_faults(outputs):
    import pandas as pd

    flagged = pd.read_csv(outputs[1] / "tables" / "flagged_runs.tsv", sep="\t")
    assert set(flagged.check) >= {"Start shift", "Pairing"}
    assert run_id(*toy_dataset.SHIFTED) in set(flagged.run)


def test_dashboard_data_is_valid_json(outputs):
    from analysis import export

    data = export.dashboard_data(outputs[1], ["toy"])
    text = json.dumps(data, allow_nan=False)
    assert json.loads(text)["overview"][0]["dataset"] == "toy"
    assert len(data["runs"]) == len(toy_runs(outputs))
