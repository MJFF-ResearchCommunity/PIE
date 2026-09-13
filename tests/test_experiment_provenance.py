"""Provenance primitives: hashing, numpy-safe JSON, manifests that can be re-verified."""
import json

import numpy as np
import pytest

from pie.experiment import provenance as prov


def test_hash_is_content_addressed_and_chunking_does_not_change_it(tmp_path, monkeypatch):
    p = tmp_path / "a.bin"
    p.write_bytes(b"x" * 3_000_000)
    digest = prov.sha256(p)
    monkeypatch.setattr(prov, "_CHUNK", 7)
    assert prov.sha256(p) == digest
    (tmp_path / "b.bin").write_bytes(b"x" * 3_000_000 + b"y")
    assert prov.sha256(tmp_path / "b.bin") != digest


def test_numpy_scalars_survive_json_and_non_finite_floats_become_null(tmp_path):
    value = {"n": np.int64(3), "auc": np.float64(0.81), "ok": np.bool_(True),
             "missing": np.float64("nan"), "arr": np.arange(2)}
    out = tmp_path / "v.json"
    prov.save_json(out, value)
    back = prov.read_json(out)
    assert back == {"n": 3, "auc": 0.81, "ok": True, "missing": None, "arr": [0, 1]}
    assert "NaN" not in out.read_text()


def test_manifest_records_inputs_and_outputs_and_verifies_afterwards(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("PATNO,y\n1,0\n")
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.csv").write_text("auc\n0.7\n")
    record = prov.write_manifest(out, inputs=[src], started=None, seed=7, include_environment=False)
    assert record["seed"] == 7
    assert record["inputs_sha256"][str(src)] == prov.sha256(src)
    assert set(record["outputs_sha256"]) == {"metrics.csv"}
    assert prov.verify_manifest(out) == []
    (out / "metrics.csv").write_text("auc\n0.9\n")       # a result edited after the fact
    assert prov.verify_manifest(out) == [str(out / "metrics.csv")]


def test_manifest_survives_a_reader_that_rejects_nan(tmp_path):
    out = tmp_path / "run"
    prov.write_manifest(out, started=None, include_environment=False, delta=np.float64("inf"))
    json.loads((out / "manifest.json").read_text(), parse_constant=lambda c: pytest.fail(c))


def test_environment_omits_packages_that_are_not_installed():
    env = prov.environment(packages=("numpy", "not_a_real_package_xyz"))
    assert "numpy" in env["packages"] and "not_a_real_package_xyz" not in env["packages"]
