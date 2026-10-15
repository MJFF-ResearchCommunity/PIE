"""The pre-registered rs-fMRI validation analysis on fully synthetic data (no real value, label or identifier)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location("analyse_fmri_validation", Path(__file__).resolve().parent.parent / "scripts" / "analyse_fmri_validation.py")
fv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fv)
FILLED = dict(fv.GATES, qcfc_median_excess=0.01, qcfc_distance_abs=0.05)     # stand-ins for the Ciric figure values
P = 50                                                                           # permutations: enough for a test


@pytest.fixture(autouse=True)
def one_blas_thread():          # small matrices: threads only add overhead (and the machine is shared)
    with threadpool_limits(1):
        yield


def test_dry_run_reports_every_protocol_and_leaves_unset_benchmarks_undecided(tmp_path):
    res = fv.main(["--synthetic", "--out", str(tmp_path), "--n-perm", "50"])
    assert set(res["protocols"]) == {"P1_TR2.4", "P1_TR2.5", "P2_TR1", "P2_TR2.5", "P2_TR1:no_notch"}
    assert json.loads((tmp_path / "synthetic_fmri_validation.json").read_text())["gates_used"]["qcfc_pct_fdr"] == 7.0
    for p, r in res["protocols"].items():
        assert r["gates"]["qcfc_median"] is None and r["validated"] is None          # Ciric figure values not yet copied
        assert ("reliability" in r["gates"]) == p.startswith("P2")
    assert res["biology"]["all"]["l"]["r"] > 0                                         # synthetic coupling tracks SBR


def test_motion_artefact_fails_qcfc_and_clean_data_passes():
    clean = fv.analyse(*fv.synthetic(n=60, motion=0.0), gates=FILLED, n_perm=P)["protocols"]
    assert all(clean[p]["validated"] for p in ("P1_TR2.4", "P1_TR2.5", "P2_TR1", "P2_TR2.5"))
    strong = fv.analyse(*fv.synthetic(n=60, motion=3.0), gates=FILLED, n_perm=P)["protocols"]["P2_TR1"]
    q = strong["qcfc"]
    assert q["percent_significant_fdr"] > 7 and q["median_excess"] > 0.1 and q["distance_dependence_r"] < -0.1
    assert strong["gates"]["qcfc_fdr"] is False and strong["gates"]["qcfc_median"] is False and strong["validated"] is False
    moderate = fv.analyse(*fv.synthetic(n=60, motion=1.0), gates=FILLED, n_perm=P)["protocols"]["P1_TR2.4"]["gates"]
    assert moderate["qcfc_fdr"] and moderate["qcfc_distance"] is False      # short-range inflation the FDR count misses


def test_halves_without_stable_individual_connectivity_fail_reliability():
    r = fv.analyse(*fv.synthetic(n=60, motion=0.0, stable=0.0), gates=FILLED, n_perm=P)["protocols"]
    rel = r["P2_TR1"]["reliability"]
    assert rel["families"]["fc_4S456"]["median_icc"] < 0.2 and not rel["families"]["fc_4S456"]["exported"]
    assert rel["edge_icc_median"] < 0.1 and rel["identification_accuracy"] < 0.2
    assert r["P2_TR1"]["gates"]["reliability"] is False and r["P2_TR1"]["validated"] is False
    assert r["P1_TR2.4"]["validated"] is True                       # PPMI-1: reliability reported, not gated
    assert rel["families"]["striatal"]["exported"]                  # run-step halves still agree


def test_throughput_counts_pipeline_failures_but_not_acquisition_exclusions():
    scans, ts, atlas = fv.synthetic(n=60, motion=0.0)
    p = scans["protocol"].eq("P1_TR2.4")
    scans.loc[p, "exclusion"] = ""
    scans.loc[scans.index[p][:20], "exclusion"] = "no_verified_reverse_pe_pair"     # acquisition: leaves the denominator
    ok = fv.analyse(scans, ts, atlas, gates=FILLED, n_perm=P)["protocols"]["P1_TR2.4"]["flow"]
    assert ok["throughput"] == 1.0 and ok["acquisition_excluded"] == 20
    scans.loc[scans.index[p][20:30], "exclusion"] = "fmriprep_error"                 # pipeline: counts against
    bad = fv.analyse(scans, ts, atlas, gates=FILLED, n_perm=P)["protocols"]["P1_TR2.4"]
    assert bad["flow"]["throughput"] == 30 / 40 and bad["gates"]["throughput"] is False


def test_icc31_is_consistency_and_selection_is_one_earliest_scan_per_participant_and_stratum():
    a = np.random.default_rng(0).normal(size=(30, 3))
    assert np.allclose(fv._icc31(a, a + 5.0), 1.0)                 # a constant offset between halves is not unreliability
    rows = [(1, "2014-01-01", "ep2d_RESTING_STATE", "I1", 2.4), (1, "2015-01-01", "ep2d_RESTING_STATE", "I2", 2.4),
            (1, "2023-01-01", "rsfMRI_AP", "I3", np.nan), (1, "2023-01-01", "rsfMRI_PA", "I4", np.nan),
            (2, "2022-05-01", "rsfMRI_RL", "I5", np.nan), (2, "2022-05-01", "rsfMRI_LR", "I6", np.nan),
            (3, "2013-01-01", "ep2d_RESTING_STATE", "I7", 2.5), (4, "2012-01-01", "DTI_gated", "I8", 2.4)]
    picked, counts = fv.select(pd.DataFrame(rows, columns=["patno", "date", "desc", "image_id", "tr"]))
    got = {(r.stratum, r.patno): r.image_ids for r in picked.itertuples()}
    assert got == {("P1_TR2.4", 1): "I1", ("P2_TR1", 1): "I3;I4", ("P2_TR2.5", 2): "I5;I6", ("P1_TR2.5", 3): "I7"}
    many = pd.DataFrame({"patno": range(300), "date": "2023-01-01", "image_id": [f"I{i}" for i in range(300)], "tr": np.nan,
                         "desc": np.where(np.arange(300) < 200, "rsfMRI_RL", "rsfMRI_RL_(no_music)")})
    first, c = fv.select(many)
    assert c["P2_TR2.5"] == {"eligible": 300, "selected": 100, "signatures": 2}
    assert first["signature"].value_counts().to_dict() == {"rsfmri_rl": 67, "rsfmri_rl_no_music_": 33}
    assert first["patno"].tolist() == fv.select(many)[0]["patno"].tolist()          # seeded: the same list every time
