"""Fetch the openly licensed Brain Explorer example bundle and write its viewer manifest.

Every file is pinned by URL and SHA-256; a file that already verifies is not
downloaded again. Two display copies are derived once: the PET averaged to 1 mm,
and DWI FA / MD / mean b0 from PIE's own tensor code. The T1 + 3-D structures entry
is written only once the FastSurfer segmentation exists; until then the command
that creates it is printed.

    venv_imaging/bin/python scripts/fetch_viewer_examples.py [--dest Imaging/examples] [--no-dwi]
    venv_imaging/bin/python -m pie.imaging.viewer serve --manifest Imaging/examples/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
S3 = "https://s3.amazonaws.com/openneuro.org"
T1 = "ds005892/sub-MJF001_T1w.nii.gz"
BOLD = "ds005892/sub-MJF001_task-rest_bold"
PET = "ds006917/sub-005/sub-005_ses-nx_trc-fluoroethylpe2i_run-02_pet"
CT = "ct_electrodes/CT_Electrodes.nii.gz"
DWI = "ds001907/sub-RC4101_ses-1/sub-RC4101_ses-1_dwi"
PET_1MM = "derived/ds006917_sub-005_trc-fluoroethylpe2i_pet_1mm.nii.gz"
DWI_MAPS = {key: f"derived/ds001907_sub-RC4101_ses-1_{key}.nii.gz" for key in ("fa", "md", "b0")}
FASTSURFER_SUBJECT = "ds005892_sub-MJF001"

# dest-relative path -> (pinned URL, SHA-256)
FILES = {
    T1: (f"{S3}/ds005892/sub-MJF001/anat/sub-MJF001_T1w.nii.gz",
         "706e49cdfb24001d842de9f1ae739b4cda430f2c452e8ae498130e4af683bc9d"),
    f"{BOLD}.nii.gz": (f"{S3}/ds005892/sub-MJF001/func/sub-MJF001_task-rest_bold.nii.gz",
                       "83495d438e4c2087ff33837a0d163dc8e032b48d4b0b0f7a716fcd2642fdb81c"),
    f"{BOLD}.json": (f"{S3}/ds005892/sub-MJF001/func/sub-MJF001_task-rest_bold.json",
                     "ee796b6d29453f1b7d312b68283a502d821f902b83f9c23c0b0e1b3ac2ffac92"),
    "ds005892/dataset_description.json": (f"{S3}/ds005892/dataset_description.json",
                                          "bf7b8980ef503cca142aa992e43db49fb25b97dfcbf22f20f6fae5854aaada75"),
    "ds005892/participants.tsv": (f"{S3}/ds005892/participants.tsv",
                                  "e82591daf4bfc74264e0b3bf261299563523bc2570e0abefda3454247c0fc8ff"),
    f"{PET}.nii.gz": (f"{S3}/ds006917/sub-005/ses-nx/pet/sub-005_ses-nx_trc-fluoroethylpe2i_run-02_pet.nii.gz",
                      "a471ef092593368b0e8dcb6a41aeb3246d28df14ce58bdc906030bcffb426f6a"),
    f"{PET}.json": (f"{S3}/ds006917/sub-005/ses-nx/pet/sub-005_ses-nx_trc-fluoroethylpe2i_run-02_pet.json",
                    "10b3846e036c72198c0ccc11fe9218525dae40d66434183d6ab24ff267439a56"),
    "ds006917/dataset_description.json": (f"{S3}/ds006917/dataset_description.json",
                                          "bf306045a3fe9555140cb09aea112f5fd062843ef50c7606da34fbe8c9b5b9b7"),
    CT: ("https://raw.githubusercontent.com/neurolabusc/niivue-images/main/CT_Electrodes.nii.gz",
         "0566942708ff451458d0fdb1f61c4611187bb5cc0efa516c8500f1ff3c142e57"),
    "ct_electrodes/Seg3DData_LICENSE": ("https://raw.githubusercontent.com/CIBC-Internal/Seg3DData/master/LICENSE",
                                        "30a7b23550b2a64eccde03066377bb72b88589c6fd2a7f875d75851cdb677ff4"),
}
DWI_FILES = {
    f"{DWI}.nii.gz": (f"{S3}/ds001907/sub-RC4101/ses-1/dwi/sub-RC4101_ses-1_dwi.nii.gz",
                      "11873180dd67f178b000b87f6729f5b3f98c93a58aa5b917f478b859bc5a9f74"),
    f"{DWI}.bval": (f"{S3}/ds001907/sub-RC4101/ses-1/dwi/sub-RC4101_ses-1_dwi.bval",
                    "a7c3993ff94f5da5c452f90c81612ceaa5a477c241563ddda18bc1ae4bd8b528"),
    f"{DWI}.bvec": (f"{S3}/ds001907/sub-RC4101/ses-1/dwi/sub-RC4101_ses-1_dwi.bvec",
                    "cf78f8d6ca11093515ae8983180e7a04a3ca766ffbabc54d0648e40515213f92"),
    f"{DWI}.json": (f"{S3}/ds001907/sub-RC4101/ses-1/dwi/sub-RC4101_ses-1_dwi.json",
                    "915f9a1a4b1176c58e18fe7b75dc0961f1098c52f39d1ef39f224ea2bf154ace"),
    "ds001907/dataset_description.json": (f"{S3}/ds001907/dataset_description.json",
                                          "413cbc0696851f1fb3d13614b869a5f8e361ae09982b27aebd2760884d42d818"),
}
DATASETS = {
    "ds005892": {"collection": "OpenNeuro ds005892", "license": "CC0",
                 "participant": "sub-MJF001: Parkinson's disease with mild cognitive impairment (PD-MCI), 68, male",
                 "citation": "Kemp AS, Eubank J, Younus Y, Galvin JE, Prior FW, Larson-Prior LJ. Resting state MRI data "
                             "from HC, PD-NC and PD-MCI cohorts. OpenNeuro ds005892 v1.0.0, doi:10.18112/openneuro.ds005892.v1.0.0. "
                             "Funded by The Michael J. Fox Foundation."},
    "ds006917": {"collection": "OpenNeuro ds006917", "license": "CC0",
                 "participant": "sub-005: healthy control",
                 "citation": "Volpi T, Toyonaga T, Khattar N, ... Carson RE. Exceptional brain PET images from the NeuroEXPLORER. "
                             "Eur J Nucl Med Mol Imaging 2025, doi:10.1007/s00259-025-07605-4. "
                             "Dataset: OpenNeuro ds006917 v1.0.2, doi:10.18112/openneuro.ds006917.v1.0.2."},
    "ds001907": {"collection": "OpenNeuro ds001907", "license": "CC0",
                 "participant": "sub-RC4101: healthy older control",
                 "citation": "Day TKM, Madyastha TM, Boord P, Askren MK, Montine TJ, Grabowski TJ. ANT: Healthy aging and "
                             "Parkinson's disease. OpenNeuro ds001907, doi:10.18112/openneuro.ds001907.v3.2.0."},
    "ct_electrodes": {"collection": "niivue-images", "license": "MIT (ct_electrodes/Seg3DData_LICENSE)",
                      "participant": "CT_Electrodes: head CT with implanted electrodes; clinical context not stated",
                      "citation": "CT_Electrodes from the Seg3DData repository, distributed in neurolabusc/niivue-images. "
                                  "Copyright (c) 2015 Scientific Computing and Imaging Institute, University of Utah."},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(dest: Path, files: dict):
    for rel, (url, digest) in files.items():
        path = dest / rel
        if path.is_file() and sha256(path) == digest:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out)
        if sha256(part) != digest:
            part.unlink()
            raise SystemExit(f"{rel}: SHA-256 mismatch for {url}; the upstream file changed. Nothing was replaced.")
        part.replace(path)
    print(f"Verified {len(files)} files under {dest}")


def save(data, affine, path: Path):
    image = nib.Nifti1Image(np.asarray(data, np.float32), affine)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    image.header.set_xyzt_units("mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("tmp-" + path.name)
    nib.save(image, temporary)
    temporary.replace(path)


def pet_1mm(dest: Path):
    """2x2x2 block mean of the 0.5 mm PET: exact mean Bq/mL at 1 mm and one-eighth of the browser memory."""
    out = dest / PET_1MM
    if out.is_file():
        return
    image = nib.load(dest / f"{PET}.nii.gz")
    data = np.asarray(image.dataobj, dtype=np.float32)
    n = [s - s % 2 for s in data.shape]
    blocks = data[:n[0], :n[1], :n[2]].reshape(n[0] // 2, 2, n[1] // 2, 2, n[2] // 2, 2).mean(axis=(1, 3, 5))
    # New voxel k covers old voxels 2k and 2k+1, so its centre is old index 2k + 0.5.
    save(blocks, image.affine @ np.array([[2, 0, 0, .5], [0, 2, 0, .5], [0, 0, 2, .5], [0, 0, 0, 1.]]), out)


def dwi_maps(dest: Path):
    """FA, MD and motion-corrected mean b0 with PIE's DWI code; the free-water fit is skipped."""
    if all((dest / p).is_file() for p in DWI_MAPS.values()):
        return
    sys.path.insert(0, str(REPO))
    from pie.imaging.dwi import fit_models, preprocess
    image = nib.load(dest / f"{DWI}.nii.gz")
    ds = {"data": np.asarray(image.dataobj, dtype=np.float32), "affine": image.affine,
          "bvals": np.loadtxt(dest / f"{DWI}.bval"), "bvecs": np.loadtxt(dest / f"{DWI}.bvec")}
    print("Fitting DWI tensors with pie.imaging.dwi (a few minutes on CPU) ...", flush=True)
    ds = preprocess(ds)
    maps = fit_models(ds, fw_mask=np.zeros(ds["mask"].shape, bool))  # empty free-water mask: FA/MD only
    for key, data in (("fa", maps["fa"]), ("md", maps["md"]), ("b0", ds["b0"])):
        save(data, image.affine, dest / DWI_MAPS[key])


def fastsurfer_command(dest: Path) -> str:
    q = lambda p: shlex.quote(str(p))
    return (f"cd {q(REPO / 'third_party/FastSurfer')} && ./run_fastsurfer.sh --t1 {q((dest / T1).resolve())} "
            f"--sid {FASTSURFER_SUBJECT} --sd {q((dest / 'fastsurfer').resolve())} --seg_only --no_cereb --no_hypothal "
            f"--no_cc --device cpu --viewagg_device cpu --threads 6 --py {q(REPO / 'venv_imaging/bin/python')}")


def sidecar(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def provenance(dataset: str, detail: str) -> str:
    d = DATASETS[dataset]
    return f"{detail} Source: {d['citation']} License: {d['license']}."


def build_manifest(dest: Path) -> dict:
    """Manifest entries for whatever part of the bundle is present under dest (paths relative to dest)."""
    visit = "Open dataset (no acquisition dates)"
    scans = []
    mri = dest / "fastsurfer" / FASTSURFER_SUBJECT / "mri"
    image = next((mri / name for name in ("orig_nu.mgz", "orig.mgz") if (mri / name).is_file()), None)
    if image and (mri / "mask.mgz").is_file() and (mri / "aparc.DKTatlas+aseg.deep.mgz").is_file():
        scans.append({"id": "mjf001-t1", "subject": "sub-MJF001", "modality": "MRI", "visit": visit,
                      "path": str(image.relative_to(dest)), "mask": str((mri / "mask.mgz").relative_to(dest)),
                      "atlas": str((mri / "aparc.DKTatlas+aseg.deep.mgz").relative_to(dest)), "atlas_name": "DKT + aseg",
                      "space": "sub-MJF001:T1-conformed", "description": "T1w · FastSurfer conformed 1 mm, brain-masked",
                      "qc": "Automated FastSurfer segmentation · not reviewed",
                      "provenance": provenance("ds005892", "T1w conformed and segmented locally with FastSurfer --seg_only (aseg + DKT); the brain mask removes the face.")})
    if (dest / f"{BOLD}.nii.gz").is_file():
        scans.append({"id": "mjf001-rest-bold", "subject": "sub-MJF001", "modality": "fMRI", "kind": "timeseries", "visit": visit,
                      "path": f"{BOLD}.nii.gz", "space": "sub-MJF001:native:bold", "units": "BOLD signal (arbitrary units)",
                      "description": "Resting-state BOLD · 200 frames, TR 2 s",
                      "qc": "Raw resting-state run · no preprocessing, not reviewed",
                      "provenance": provenance("ds005892", "Unprocessed resting-state BOLD as distributed (dcm2niix)."),
                      "metadata": {"example": True, "run_role": "Resting-state BOLD run", "sidecar": sidecar(dest / f"{BOLD}.json")}})
    if (dest / PET_1MM).is_file():
        scans.append({"id": "sub005-fepe2i-pet", "subject": "sub-005", "modality": "PET", "visit": visit, "path": PET_1MM,
                      "space": "sub-005:native:pet", "tracer": "[18F]FE-PE2I",
                      "units": "Bq/mL (decay-corrected; 40–90 min after injection)",
                      "description": "[18F]FE-PE2I dopamine-transporter PET · 40–90 min",
                      "qc": "Healthy control · native PET space, not registered to any MRI",
                      "provenance": provenance("ds006917", "NeuroEXPLORER (UIH uNeuroX) list-mode PET, 3D OSEM-PSF-TOF, frame 2392–5392 s after injection; the 0.5 mm image is averaged 2x2x2 to 1 mm by fetch_viewer_examples.py (mean Bq/mL preserved)."),
                      "metadata": {"sidecar": sidecar(dest / f"{PET}.json")}})
    if (dest / CT).is_file():
        scans.append({"id": "ct-electrodes", "subject": "CT_Electrodes", "modality": "CT", "visit": visit, "path": CT,
                      "space": "CT_Electrodes:native", "units": "HU", "description": "Head CT with implanted electrodes",
                      "qc": "Example image · clinical context not stated",
                      "provenance": provenance("ct_electrodes", "Head CT as distributed.")})
    if all((dest / p).is_file() for p in DWI_MAPS.values()):
        scans.append({"id": "rc4101-dti", "subject": "sub-RC4101", "modality": "DTI", "visit": visit,
                      "path": DWI_MAPS["fa"], "anatomy": DWI_MAPS["b0"], "space": "sub-RC4101:DWI", "units": "FA (dimensionless)",
                      "extra": [{"name": "Mean diffusivity", "path": DWI_MAPS["md"], "units": "mm²/s", "key": "md"}],
                      "description": "Fractional anisotropy · native diffusion space", "qc": "PIE tensor fit · not reviewed",
                      "provenance": provenance("ds001907", "DWI with 1 b0 + 128 directions at b=1000 (per the .bval; the sidecar SeriesDescription says b3000) -> pie.imaging.dwi.preprocess (median-Otsu mask, rigid motion correction to the mean b0) and fit_models (WLS tensor, b <= 1000); free water not fitted.")})
    people = {"sub-MJF001": ("ds005892", {"group": "PD-MCI", "cohort": "PD-MCI", "sex": "M", "age_at_scan": "68"}),
              "sub-005": ("ds006917", {"group": "Healthy control", "cohort": "Healthy control"}),
              "sub-RC4101": ("ds001907", {"group": "Healthy control", "cohort": "Healthy control"}),
              "CT_Electrodes": ("ct_electrodes", {"group": "Example CT", "cohort": "Example CT"})}
    used = {s["subject"] for s in scans}
    subjects = [{"id": sid, "collection": DATASETS[key]["collection"], **fields} for sid, (key, fields) in people.items() if sid in used]
    return {"version": 1, "subjects": subjects, "scans": scans}


def write_outputs(dest: Path) -> dict:
    doc = build_manifest(dest)
    (dest / "manifest.json").write_text(json.dumps(doc, indent=2))
    lines = ["# Brain Explorer example data: attribution", "",
             "Written by `scripts/fetch_viewer_examples.py`. Keep this file with the data.", ""]
    for key, d in DATASETS.items():
        lines += [f"## {d['collection']} (`{key}/`)", "", f"- Participant: {d['participant']}",
                  f"- License: {d['license']}", f"- Cite: {d['citation']}", ""]
    lines += ["## Derived files", "",
              f"- `{PET_1MM}`: 2x2x2 block mean of the ds006917 PET (0.5 mm to 1 mm).",
              "- `derived/ds001907_*`: FA, MD and mean b0 from pie.imaging.dwi (display copies).",
              f"- `fastsurfer/{FASTSURFER_SUBJECT}/`: local FastSurfer segmentation of the ds005892 T1 "
              "(FastSurfer, Apache-2.0). Derived files carry their source dataset's license.", ""]
    (dest / "ATTRIBUTION.md").write_text("\n".join(lines))
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=REPO / "Imaging/examples")
    parser.add_argument("--no-dwi", action="store_true",
                        help="Skip the diffusion example (67 MB download plus a few minutes of tensor fitting)")
    args = parser.parse_args()
    dest = args.dest.resolve()
    fetch(dest, FILES if args.no_dwi else FILES | DWI_FILES)
    pet_1mm(dest)
    if not args.no_dwi:
        dwi_maps(dest)
    doc = write_outputs(dest)
    print(f"Wrote {dest / 'manifest.json'} ({len(doc['scans'])} scans) and {dest / 'ATTRIBUTION.md'}")
    if not any(s["id"] == "mjf001-t1" for s in doc["scans"]):
        print("The T1 + 3-D structures entry needs a FastSurfer segmentation (CPU, roughly 15-30 min). Run:\n  "
              + fastsurfer_command(dest) + "\nthen run this script again.")
    print(f"Serve: venv_imaging/bin/python -m pie.imaging.viewer serve --manifest {dest / 'manifest.json'}")


if __name__ == "__main__":
    main()
