"""Convert explicitly selected local IDA BOLD series; preserve every source frame.

Archives are read-only. Do not infer BOLD from IDA's modality checkbox, concatenate
opposite phase-encoding runs, or silently choose the largest converter output.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pie.imaging.convert import DCM2NIIX
from pie.imaging.viewer.catalog import iso_date, merge_manifest, read_manifest, source_fingerprint


def bold_description(description):
    return bool(re.search(r"fmri|bold|resting", description, re.I)) and not bool(
        re.search(r"dti|diffusion|revb0|localizer|gre.?mt", description, re.I))


def validate_bold(path, metadata):
    image = nib.load(path)
    if len(image.shape) != 4 or image.shape[3] < 2:
        raise ValueError("BOLD example must contain a complete 4D time series")
    if not np.all(np.isfinite(image.affine)) or abs(np.linalg.det(image.affine[:3, :3])) < 1e-9:
        raise ValueError("Unusable spatial geometry")
    tr = metadata.get("RepetitionTime")
    if not isinstance(tr, (int, float)) or not np.isfinite(tr) or tr <= 0:
        raise ValueError("A positive RepetitionTime in the converter JSON is required")
    if image.header.get_xyzt_units()[1] != "sec" or not np.isclose(image.header.get_zooms()[3], tr, rtol=1e-4):
        raise ValueError("NIfTI frame timing and JSON RepetitionTime disagree")
    return image


def prepare(archives, collection, images, output, max_bytes=3_000_000_000):
    with collection.open(encoding="utf-8-sig") as f:
        rows = {r["Image Data ID"]: r for r in csv.DictReader(f)}
    if not images or len(images) != len(set(images)):
        raise ValueError("Select unique explicit image IDs")
    for image_id in images:
        row = rows.get(image_id)
        if not re.fullmatch(r"I\d+", image_id) or not row or not bold_description(row["Description"]) or not iso_date(row["Acq Date"]):
            raise ValueError(f"{image_id}: not a dated BOLD candidate in the supplied collection")
    selected, inventories, all_ids = {}, [], set()
    for archive in archives:
        ids, overlaps = set(), 0
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                parts = member.filename.split("/")
                if len(parts) != 6 or not re.fullmatch(r"I\d+", parts[4]) or member.is_dir():
                    continue
                image_id = parts[4]
                ids.add(image_id)
                if image_id not in images:
                    continue
                row = rows[image_id]
                if parts[1] != row["Subject"] or not re.fullmatch(r"\d+", parts[1]):
                    raise ValueError("Archive participant differs from collection")
                prefix = "/".join(parts[:5]) + "/"
                item = selected.setdefault(image_id, {"archive": archive, "prefix": prefix, "members": []})
                if item["archive"] != archive or item["prefix"] != prefix:
                    raise ValueError(f"{image_id}: ambiguous duplicate across archives or sessions")
                item["members"].append((member.filename, member.file_size, member.CRC))
        overlaps = len(ids & all_ids)
        all_ids |= ids
        inventories.append({"archive": str(archive), "bytes": archive.stat().st_size, "series": len(ids), "overlapping_previous": overlaps})
    if set(selected) != set(images):
        raise ValueError(f"Selected series missing: {set(images) - set(selected)}")
    if sum(size for item in selected.values() for _, size, _ in item["members"]) > max_bytes:
        raise ValueError("Explicit selection exceeds the uncompressed extraction budget")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    read_manifest(manifest_path)  # refuse an unusable manifest before converting anything
    converter = subprocess.run([DCM2NIIX, "--version"], capture_output=True, text=True).stdout.strip()
    scans, subjects = [], []
    for image_id in images:
        row, source = rows[image_id], selected[image_id]
        signature = source_fingerprint(source["archive"], source["members"])
        folder = output / row["Subject"] / image_id
        nifti, sidecar = folder / f"{image_id}_bold.nii.gz", folder / f"{image_id}_bold.json"
        receipt = folder / "conversion.json"
        if folder.exists() and not receipt.exists():
            raise ValueError(f"Incomplete or unrelated output exists: {folder}. Inspect before retrying.")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["source_fingerprint"] != signature:
                raise ValueError("Source archive contents changed; refusing to overwrite a converted example")
        else:
            with tempfile.TemporaryDirectory(prefix="bold-convert-", dir=output) as temporary:
                temp = Path(temporary)
                dicom, converted = temp / "dicom", temp / "converted"
                dicom.mkdir(); converted.mkdir()
                with zipfile.ZipFile(source["archive"]) as z:
                    for index, (name, _, _) in enumerate(source["members"]):
                        with z.open(name) as src, (dicom / f"{index:08d}.dcm").open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                result = subprocess.run([DCM2NIIX, "-z", "y", "-b", "y", "-ba", "y", "-m", "n", "-f", f"{image_id}_%s", "-o", str(converted), str(dicom)], capture_output=True, text=True)
                paths = list(converted.glob("*.nii.gz"))
                if result.returncode or len(paths) != 1:
                    raise ValueError(f"{image_id}: expected one unsplit BOLD output; got {len(paths)}. {result.stdout[-2000:]} {result.stderr[-500:]}")
                js = paths[0].with_suffix("").with_suffix(".json")
                meta = json.loads(js.read_text())
                validate_bold(paths[0], meta)
                staging = temp / "ready"
                staging.mkdir()
                shutil.move(paths[0], staging / nifti.name)
                shutil.move(js, staging / sidecar.name)
                (staging / receipt.name).write_text(json.dumps({"source_fingerprint": signature, "archive": str(source["archive"]), "prefix": source["prefix"], "dicom_files": len(source["members"]), "converter": converter, "log": result.stdout + result.stderr}, indent=2))
                folder.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(folder)
        meta = json.loads(sidecar.read_text())
        image = validate_bold(nifti, meta)
        short = image.shape[3] < 30
        role = "Short EPI reference candidate" if short else "Resting-state BOLD run"
        scans.append({"id": f"bold-{image_id}", "subject": row["Subject"], "modality": "fMRI", "kind": "timeseries",
                      "date": iso_date(row["Acq Date"]), "visit": row["Visit"], "description": row["Description"],
                      "path": str(nifti.resolve()), "space": f"sub-{row['Subject']}:native:{image_id}", "units": "BOLD signal (arbitrary units)",
                      "registration": "native", "qc": f"{role} · preprocessing and quality not reviewed",
                      "provenance": "Local IDA DICOM → dcm2niix; all converted frames retained in native EPI geometry. No motion, slice-timing or susceptibility correction, denoising, T1 registration, activation model, or connectivity analysis applied.",
                      "metadata": {"image_id": image_id, "archive": str(source["archive"]), "source_fingerprint": signature,
                                   "run_role": role, "short_reference": short, "example": True,
                                   "manufacturer": meta.get("Manufacturer"), "model": meta.get("ManufacturersModelName"),
                                   "protocol": meta.get("SeriesDescription"), "field_t": meta.get("MagneticFieldStrength"), "sidecar": meta}})
        subjects.append({"id": row["Subject"], "collection": "PPMI", "group": row["Group"], "cohort": row["Group"], "sex": row["Sex"], "age_at_scan": row["Age"]})
        print(f"{image_id}: {row['Subject']} {image.shape}, TR {meta['RepetitionTime']} s, PE {meta.get('PhaseEncodingDirection', 'unknown')} · {role}", flush=True)
    merge_manifest(manifest_path, subjects, scans)
    report = {"archives": inventories, "collection": str(collection), "collection_series": len(rows), "distinct_archive_series": len(all_ids), "missing_ids": sorted(set(rows)-all_ids), "extra_ids": sorted(all_ids-set(rows)), "selected": images, "note": "Central-directory reconciliation is not a pixel-data QC review. Selected DICOM members passed ZIP CRC during extraction; full archives were not decompressed."}
    (output / "fmri_archive_inventory.json").write_text(json.dumps(report, indent=2))
    print(f"Added {len(scans)} examples to {manifest_path}; original archives unchanged", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", nargs="+", type=Path, required=True)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.archives, a.collection, a.images, a.output)
