"""Viewer helper scripts: shared collection manifest, launcher arguments, plan output."""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pie.imaging.viewer import __main__, server
from pie.imaging.viewer.catalog import Catalog, merge_manifest
from scripts import fetch_viewer_examples as examples
from scripts.prepare_viewer_followup import prepare as prepare_followup

REPO = Path(__file__).resolve().parents[1]


def nifti(path, shape):
    image = nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4))
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, path)
    return path


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def followup_inputs(tmp_path):
    archive = tmp_path / "collection.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for n, image_id in enumerate(("I1", "I2"), 1):
            z.writestr(f"PPMI/P001/T1_MPRAGE/2000-0{n}-01_00_00_00.0/{image_id}/slice.dcm", b"synthetic")
    table = tmp_path / "collection.csv"
    with table.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Data ID", "Subject", "Group", "Sex", "Visit", "Description", "Acq Date"])
        writer.writerow(["I1", "P001", "Synthetic", "F", "BL", "T1 MPRAGE", "1/01/2000"])
        writer.writerow(["I2", "P001", "Synthetic", "F", "V04", "T1 MPRAGE", "1/01/2001"])
    return archive, table


def fake_convert(archive, prefix, patno, image_id, out):
    folder = Path(out) / patno
    folder.mkdir(parents=True, exist_ok=True)
    sidecar = folder / f"{image_id}_T1w.json"
    sidecar.write_text(json.dumps({"Manufacturer": "Synthetic"}))
    return {"nifti": str(nifti(folder / f"{image_id}_T1w.nii.gz", (4, 4, 4))), "sidecar": str(sidecar)}


def bold_entry(tmp_path, fingerprint="bold-source"):
    return {"id": "bold-I9", "subject": "P002", "modality": "fMRI", "kind": "timeseries",
            "path": str(nifti(tmp_path / "bold.nii.gz", (4, 4, 4, 3))), "metadata": {"source_fingerprint": fingerprint}}


def test_followup_and_fmri_share_one_manifest_in_either_order(tmp_path, followup_inputs):
    archive, table = followup_inputs
    for order in ("fmri-first", "followup-first"):
        output = tmp_path / order
        manifest = output / "manifest.json"
        if order == "fmri-first":
            merge_manifest(manifest, [{"id": "P002"}], [bold_entry(tmp_path)])
        prepare_followup(archive, table, "P001", ["I1", "I2"], output, convert=fake_convert)
        if order == "followup-first":
            merge_manifest(manifest, [{"id": "P002"}], [bold_entry(tmp_path)])
        catalog = Catalog(tmp_path / "repo", manifest=manifest)
        assert set(catalog.scans) == {"local-I1", "local-I2", "bold-I9"}
        assert set(catalog.subjects) == {"P001", "P002"}
    # Re-running the same pair refreshes it; reusing an ID for another source is refused.
    prepare_followup(archive, table, "P001", ["I1", "I2"], output, convert=fake_convert)
    with pytest.raises(ValueError, match="different source"):
        merge_manifest(manifest, [], [bold_entry(tmp_path, "another-source")])


def test_launcher_builds_the_sample_plan_with_the_callers_arguments(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "brain-viewer/node_modules").mkdir(parents=True)
    shutil.copy(REPO / "scripts/run_brain_viewer.sh", root / "scripts")
    calls = tmp_path / "calls.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = executable(tmp_path / "python", f'#!/bin/sh\necho "$@" >> "{calls}"\n')
    executable(fake_bin / "npm", "#!/bin/sh\nexit 0\n")
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}", "PIE_VIEWER_PYTHON": str(python)}
    subprocess.run(["bash", str(root / "scripts/run_brain_viewer.sh"), "--ppmi-dir", "/data/ppmi", "--output", "/data/plan"],
                   env=env, check=True, capture_output=True)
    lines = calls.read_text().splitlines()
    assert "-m pie.imaging.viewer sample-plan --if-missing --ppmi-dir /data/ppmi --output /data/plan" in lines
    assert lines[-1] == "-m pie.imaging.viewer serve --ppmi-dir /data/ppmi --output /data/plan"


def test_sample_plan_output_is_used_by_cli_and_api(tmp_path, monkeypatch):
    output = tmp_path / "plans"
    argv = ["viewer", "sample-plan", "--repo", str(tmp_path), "--ppmi-dir", str(tmp_path / "ppmi"), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    __main__.main()
    plan = json.loads((output / "plan.json").read_text())
    assert any("Participant_Status" in w for w in plan["warnings"])
    (output / "plan.json").write_text(json.dumps({"marker": "kept"}))
    monkeypatch.setattr(sys, "argv", argv + ["--if-missing"])
    __main__.main()
    response = TestClient(server.create_app(tmp_path, sample_plan=output)).get("/api/sample-plan").json()
    assert response["available"] and response["marker"] == "kept" and response["download_guide"] is False


def test_open_example_manifest_loads_and_adds_t1_only_after_segmentation(tmp_path):
    """Tiny synthetic stand-ins at the bundle's paths; no downloads."""
    dest = tmp_path / "examples"
    for rel, shape in [(f"{examples.BOLD}.nii.gz", (4, 4, 4, 3)), (examples.PET_1MM, (4, 4, 4)), (examples.CT, (4, 4, 4)),
                       *[(p, (4, 4, 4)) for p in examples.DWI_MAPS.values()]]:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        nifti(dest / rel, shape)
    doc = examples.write_outputs(dest)
    assert "mjf001-t1" not in {s["id"] for s in doc["scans"]}
    assert "--seg_only" in examples.fastsurfer_command(dest)
    mri = dest / "fastsurfer" / examples.FASTSURFER_SUBJECT / "mri"
    mri.mkdir(parents=True)
    for name in ("orig.mgz", "mask.mgz", "aparc.DKTatlas+aseg.deep.mgz"):
        nib.save(nib.MGHImage(np.ones((4, 4, 4), np.float32), np.eye(4)), mri / name)
    examples.write_outputs(dest)
    catalog = Catalog(tmp_path / "repo", manifest=dest / "manifest.json")
    assert set(catalog.scans) == {"mjf001-t1", "mjf001-rest-bold", "sub005-fepe2i-pet", "ct-electrodes", "rc4101-dti"}
    assert catalog.scans["mjf001-t1"].atlas_lut is not None  # DKT + aseg falls back to the bundled lookup table
    assert catalog.subjects["sub-MJF001"]["collection"] == "OpenNeuro ds005892"
    pet = catalog.scans["sub005-fepe2i-pet"]
    assert pet.modality == "PET" and pet.tracer == "[18F]FE-PE2I" and "CC0" in pet.provenance
    assert "Scientific Computing and Imaging Institute" in (dest / "ATTRIBUTION.md").read_text()


# LONI image IDs, or a number of 3+ digits right after a participant-like word.
EMBEDDED_ID = re.compile(r"\bI\d{5,}\b|\b(?:participants?|examples?|PATNO|PPMI)\s*(?:<[^>]*>\s*)?\d{3,}", re.I)


def test_viewer_sources_do_not_embed_participant_or_image_ids():
    """On-screen IDs must come from the catalogue and the sample plan at runtime."""
    sources = [*REPO.glob("brain-viewer/src/*.ts"), *REPO.glob("brain-viewer/src/*.tsx"),
               *REPO.glob("pie/imaging/viewer/*.py"), *REPO.glob("scripts/*viewer*")]
    hits = [f"{p.relative_to(REPO)}: {m[0]}" for p in sources for m in EMBEDDED_ID.finditer(p.read_text())]
    assert sources and hits == []


def ppmi_tables(tmp_path):
    """Synthetic tables: one participant with MRI + DTI candidates, one with MRI only."""
    ppmi = tmp_path / "ppmi"
    (ppmi / "_Subject_Characteristics").mkdir(parents=True)
    (ppmi / "_Subject_Characteristics/Participant_Status_01Jan2000.csv").write_text(
        "PATNO,COHORT_DEFINITION\n101,Synthetic cohort\n102,Synthetic cohort\n")
    (ppmi / "Imaging").mkdir()
    (ppmi / "Imaging/MRI_Acquisition_Metadata_01Jan2000.csv").write_text(
        "PATNO,EVENT_ID,MRI_SCAN_DATE,MRI_SEQ_DTI,MRI_SEQ_RS\n"
        "101,BL,2000-01-01,Yes,No\n101,V04,2001-01-01,Yes,No\n102,BL,2000-02-02,No,No\n")
    return ppmi


def test_sample_plan_next_bundle_is_derived_from_local_tables(tmp_path):
    from pie.imaging.viewer.sample_plan import build_plan
    output = tmp_path / "plan"
    build_plan(tmp_path / "repo", ppmi_tables(tmp_path), output)
    bundle = json.loads((output / "plan.json").read_text())["next_bundle"]
    assert [b["subject"] for b in bundle] == ["101", "102"]  # most incomplete first
    assert [r["modality"] for r in bundle[0]["requests"]] == ["MRI", "DTI"]
    assert bundle[0]["requests"][0]["dates"] == ["2000-01-01", "2001-01-01"]
    assert bundle[0]["cohort"] == "Synthetic cohort" and bundle[0]["local_modalities"] == []
    assert "Next bundle" in (output / "DOWNLOAD_PLAN.md").read_text()
    # No PPMI tables at all: nothing to suggest, no crash.
    build_plan(tmp_path / "repo", tmp_path / "absent", output)
    assert json.loads((output / "plan.json").read_text())["next_bundle"] == []
