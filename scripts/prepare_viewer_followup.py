"""Prepare a bounded, explicit MRI pair from an existing local archive.

Raw archives remain untouched; conversion products and metadata are local and
gitignored. Nothing is called registered, skull stripped, or segmented here.
Scans are merged into <output>/manifest.json by ID, so this script and
prepare_viewer_fmri.py can share one collection folder in either order.
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pie.imaging.convert import convert_series
from pie.imaging.viewer.catalog import iso_date, merge_manifest, read_manifest, source_fingerprint


def prepare(archive, collection, subject, images, output, convert=convert_series):
    with collection.open(encoding="utf-8-sig") as f:
        rows = {r["Image Data ID"]: r for r in csv.DictReader(f) if r["Image Data ID"] in images}
    if len(images) != 2 or len(rows) != 2 or any(r["Subject"] != subject or "T1" not in r["Description"] for r in rows.values()):
        raise ValueError("Select two explicit T1 series belonging to the same participant")
    if len({iso_date(r["Acq Date"]) for r in rows.values()}) != 2:
        raise ValueError("The pair must contain two distinct acquisition dates")
    series = {}
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts = member.filename.split("/")
            if len(parts) != 6 or parts[1] != subject or parts[4] not in rows or member.is_dir():
                continue
            prefix = "/".join(parts[:5]) + "/"
            item = series.setdefault(parts[4], {"prefix": prefix, "members": []})
            if item["prefix"] != prefix:
                raise ValueError(f"{parts[4]}: series appears under more than one session folder")
            item["members"].append((member.filename, member.file_size, member.CRC))
    if set(series) != set(rows):
        raise ValueError("One or both requested series are not present in the completed archive")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    read_manifest(manifest)  # refuse an unusable manifest before converting anything
    scans = []
    for image_id in images:
        row = rows[image_id]
        converted = convert(archive, series[image_id]["prefix"], subject, image_id, output)
        metadata = json.loads(Path(converted["sidecar"]).read_text())
        scans.append({"id": f"local-{image_id}", "subject": subject, "modality": "MRI",
                      "date": iso_date(row["Acq Date"]), "visit": row["Visit"], "description": row["Description"],
                      "path": str(Path(converted["nifti"]).resolve()), "space": f"sub-{subject}:native:{image_id}",
                      "units": "arbitrary intensity", "registration": "native",
                      "qc": "Converted native head MRI · skull stripping and segmentation not performed",
                      "provenance": "Existing local IDA DICOM series → dcm2niix. No registration, normalization, brain extraction, or quantitative interpretation.",
                      "metadata": {"image_id": image_id, "archive": str(archive),
                                   "source_fingerprint": source_fingerprint(archive, series[image_id]["members"]),
                                   "manufacturer": metadata.get("Manufacturer"),
                                   "model": metadata.get("ManufacturersModelName"), "field_t": metadata.get("MagneticFieldStrength"),
                                   "protocol": metadata.get("ProtocolName"), "sidecar": metadata}})
    first = rows[images[0]]
    merge_manifest(manifest, [{"id": subject, "collection": "PPMI", "group": first["Group"],
                               "cohort": first["Group"], "sex": first["Sex"]}], scans)
    print(f"Prepared {len(scans)} native MRI scans: {manifest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--images", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.collection, args.subject, args.images, args.output)


if __name__ == "__main__":
    main()
