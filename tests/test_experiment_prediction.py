"""The leakage guarantees the engine exists to provide, on synthetic data only."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pie.experiment import prediction as pred


def frame_fixture(n=90):
    rng = np.random.default_rng(12)
    d = pd.DataFrame({"PATNO": np.arange(n), "age_at_scan": rng.uniform(40, 80, n),
                      "sex_male": rng.integers(0, 2, n), "saa_prodromal": np.arange(n) % 2,
                      "scanner_batch": np.where(np.arange(n) % 3, "GE", "Siemens"),
                      "MaskVol": rng.normal(1_500_000, 100_000, n), "image": rng.normal(size=n),
                      "empty": np.nan})
    for c in pred.COVARIATES[2:]:
        d[c] = rng.integers(0, 2, n)
    return d


def test_preprocessing_uses_only_fit_participant_statistics():
    d = frame_fixture()
    train, test = d.iloc[:60].copy(), d.iloc[60:].copy()
    test["only_test"] = 999
    train["only_test"] = np.nan
    design = pred.ImageDesign(["image", "only_test", "empty"], True, False, ["scanner_batch"], "MaskVol").fit(train)
    assert design.cols == ["image"] and design.fit_ids == tuple(range(60))
    before = design.imp.statistics_.copy()
    test["image"] = 1e12
    test["scanner_batch"] = "unseen"
    assert np.isfinite(design.transform(test)).all()
    np.testing.assert_array_equal(design.imp.statistics_, before)


def test_nigral_measures_are_not_residualized_for_intracranial_volume():
    d = frame_fixture()
    d["new_nm_contrast"] = d.image
    d["new_dti_sn_l_fa"] = d.image
    design = pred.ImageDesign(["image", "new_nm_contrast", "new_dti_sn_l_fa"], True, False,
                              ["scanner_batch"], "MaskVol").fit(d)
    blocks = {tuple(np.flatnonzero(which)): nuisance.cols for which, nuisance, _ in design.regressions}
    assert "MaskVol" in blocks[(0,)] and "MaskVol" not in blocks[(1, 2)]


def test_nested_predictions_cannot_depend_on_outer_test_labels(monkeypatch):
    d = frame_fixture(60)
    train, test = np.arange(48), np.arange(48, 60)
    monkeypatch.setattr(pred, "candidate_grid", lambda families: [pred.Candidate("baseline"), pred.Candidate("image")])
    first, audit = pred.nested_fold(d, train, test, {"image": ["image"]}, ["scanner_batch"], "MaskVol", False, 1)
    d.loc[test, "saa_prodromal"] = 1 - d.loc[test, "saa_prodromal"]
    second, _ = pred.nested_fold(d, train, test, {"image": ["image"]}, ["scanner_batch"], "MaskVol", False, 1)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    for split in audit["inner_audit"]:
        assert not set(split["fit_patnos"]) & set(split["validation_patnos"])
        assert not set(split["fit_patnos"]) & set(test)


def test_a_participant_on_both_sides_of_the_split_is_refused():
    d = frame_fixture(40)
    d.loc[39, "PATNO"] = d.loc[0, "PATNO"]
    with pytest.raises(ValueError, match="Participant leakage"):
        pred.nested_fold(d, np.arange(30), np.arange(30, 40), {"image": ["image"]}, [], None, False, 1)


def test_paired_bootstrap_keeps_repeated_people_together():
    d = pd.DataFrame({"PATNO": np.tile(np.arange(20), 3), "repeat": np.repeat(np.arange(3), 20),
                      "y": np.tile(np.arange(20) % 2, 3), "p_baseline": .5,
                      "p_selected": np.tile(np.where(np.arange(20) % 2, .8, .2), 3)})
    m = pred.paired_metrics(d, n_boot=20).set_index("model")
    assert m.loc["selected", "n"] == 20 and m.loc["selected", "delta_auc"] == .5
    assert m.loc["baseline", "delta_ci95_low"] == 0 and m.loc["baseline", "delta_ci95_high"] == 0


def test_nested_engine_can_recover_an_injected_training_signal(monkeypatch):
    d = frame_fixture(120)
    d["image"] = 4 * (2 * d.saa_prodromal - 1) + np.random.default_rng(7).normal(size=len(d))
    monkeypatch.setattr(pred, "candidate_grid", lambda families: [pred.Candidate("baseline"), pred.Candidate("image")])
    predictions, audit = pred.nested_fold(d, np.arange(90), np.arange(90, 120), {"image": ["image"]}, [], None, False, 17)
    assert audit["selections"]["selected"]["candidate"]["family"] == "image"
    assert roc_auc_score(d.iloc[90:].saa_prodromal, predictions["selected"]) > .98


def test_the_outcome_column_is_configurable_and_defaults_to_the_study_name(monkeypatch):
    d = frame_fixture(60).rename(columns={"saa_prodromal": "converted"})
    monkeypatch.setattr(pred, "candidate_grid", lambda families: [pred.Candidate("baseline")])
    predictions, _ = pred.nested_fold(d, np.arange(48), np.arange(48, 60), {}, [], None, False, 1,
                                      outcome="converted")
    assert len(predictions["baseline"]) == 12


def test_string_participant_ids_are_supported_and_integer_ids_stay_int(monkeypatch):
    monkeypatch.setattr(pred, "candidate_grid", lambda families: [pred.Candidate("baseline")])
    d = frame_fixture(60)
    d["PATNO"] = [f"P{i:03d}" for i in range(60)]
    _, audit = pred.nested_fold(d, np.arange(48), np.arange(48, 60), {}, [], None, False, 1)
    assert all(isinstance(p, str) for p in audit["inner_audit"][0]["fit_patnos"])
    _, audit = pred.nested_fold(frame_fixture(60), np.arange(48), np.arange(48, 60), {}, [], None, False, 1)
    assert all(type(p) is int for p in audit["inner_audit"][0]["fit_patnos"])


def test_a_final_refit_that_fails_falls_back_and_is_recorded(monkeypatch):
    from dataclasses import asdict
    from sklearn.exceptions import ConvergenceWarning
    d = frame_fixture(120)
    d["image"] = 4 * (2 * d.saa_prodromal - 1) + np.random.default_rng(7).normal(size=len(d))
    grid = [pred.Candidate("baseline"), pred.Candidate("image"), pred.Candidate("image", strength=1.0)]
    monkeypatch.setattr(pred, "candidate_grid", lambda families: grid)
    real, failed = pred.fit_checked, []

    def flaky(candidate, x, y, seed):
        # Converges in every inner fold, fails once on the full training partition.
        if candidate.family == "image" and len(y) == 90 and not failed:
            failed.append(grid.index(candidate))
            raise ConvergenceWarning("synthetic non-convergence")
        return real(candidate, x, y, seed)
    monkeypatch.setattr(pred, "fit_checked", flaky)
    predictions, audit = pred.nested_fold(d, np.arange(90), np.arange(90, 120), {"image": ["image"]},
                                          [], None, False, 17)
    final = [f for f in audit["failures"] if f["stage"] == "final_refit"]
    assert [f["candidate"] for f in final] == failed
    assert audit["selections"]["image"]["candidate"] != asdict(grid[failed[0]])
    assert audit["selections"]["image"]["candidate"]["family"] == "image"
    assert len(predictions["image"]) == 30


def test_connectivity_features_are_not_residualized_for_intracranial_volume():
    d = frame_fixture()
    d["fmri_dmn__salience"] = d.image
    design = pred.ImageDesign(["image", "fmri_dmn__salience"], True, False, ["scanner_batch"], "MaskVol").fit(d)
    blocks = {tuple(np.flatnonzero(which)): nuisance.cols for which, nuisance, _ in design.regressions}
    assert "MaskVol" in blocks[(0,)] and "MaskVol" not in blocks[(1,)]
