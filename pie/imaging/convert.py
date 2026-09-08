"""
convert.py — extract one DICOM series from a LONI zip and convert it to NIfTI with dcm2niix.

Keeps <out_dir>/<PATNO>/<IMAGEID>_T1w.nii.gz and the dcm2niix JSON sidecar (scanner
metadata used later as ComBat batch covariates). The DICOM files are deleted after
conversion; the zip remains the raw archive.
"""

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

DCM2NIIX = str(Path(__file__).resolve().parents[2] / "venv_imaging" / "bin" / "dcm2niix")

# JSON sidecar fields worth keeping as per-scan metadata.
SIDECAR_FIELDS = ["Manufacturer", "ManufacturersModelName", "MagneticFieldStrength", "SoftwareVersions",
                  "InstitutionName", "StationName", "DeviceSerialNumber", "SeriesDescription", "ProtocolName",
                  "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "SliceThickness",
                  "AcquisitionMatrixPE", "ReconMatrixPE", "ParallelReductionFactorInPlane", "AcquisitionTime"]


def convert_series(zip_path, member_prefix, patno, image_id, out_dir, dcm2niix=DCM2NIIX):
    """Convert one series. Returns dict(nifti=..., sidecar=..., **metadata). Idempotent."""
    subj_dir = Path(out_dir) / str(patno)
    nifti = subj_dir / f"{image_id}_T1w.nii.gz"
    sidecar = subj_dir / f"{image_id}_T1w.json"
    if nifti.exists() and sidecar.exists():
        return _row(nifti, sidecar)
    subj_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            members = [n for n in z.namelist() if n.startswith(member_prefix) and not n.endswith("/")]
            if not members:
                raise FileNotFoundError(f"{member_prefix} not in {zip_path}")
            for n in members:
                with z.open(n) as src, open(tmp / Path(n).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        out = tmp / "out"
        out.mkdir()
        r = subprocess.run([dcm2niix, "-z", "y", "-b", "y", "-f", f"{image_id}_%s", "-o", str(out), str(tmp)],
                           capture_output=True, text=True)
        niftis = sorted(out.glob("*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
        if r.returncode != 0 or not niftis:
            raise RuntimeError(f"dcm2niix failed for {member_prefix}: {r.stdout[-500:]} {r.stderr[-500:]}")
        best = niftis[0]  # largest volume if dcm2niix split the series (e.g. echoes / derived images)
        shutil.move(str(best), nifti)
        js = best.with_suffix("").with_suffix(".json")
        if js.exists():
            shutil.move(str(js), sidecar)
        else:
            sidecar.write_text("{}")
    return _row(nifti, sidecar)


def _row(nifti, sidecar):
    meta = json.loads(Path(sidecar).read_text() or "{}")
    row = {"nifti": str(nifti), "sidecar": str(sidecar)}
    for k in SIDECAR_FIELDS:
        v = meta.get(k)
        row[k] = ";".join(map(str, v)) if isinstance(v, list) else v
    return row
