"""Prepare a bounded, explicit MRI pair from an existing local archive.

Raw archives remain untouched; conversion products and metadata are local and
gitignored. Nothing is called registered, skull stripped, or segmented here.
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pie.imaging.convert import convert_series
from pie.imaging.viewer.catalog import iso_date


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--images", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.collection.open(encoding="utf-8-sig") as f:
        rows = {r["Image Data ID"]: r for r in csv.DictReader(f) if r["Image Data ID"] in args.images}
    if len(rows) != 2 or any(r["Subject"] != args.subject or "T1" not in r["Description"] for r in rows.values()):
        raise ValueError("Select two explicit T1 series belonging to the same participant")
    if len({iso_date(r["Acq Date"]) for r in rows.values()}) != 2:
        raise ValueError("The pair must contain two distinct acquisition dates")
    with zipfile.ZipFile(args.archive) as z:
        prefixes = {p[4]: "/".join(p[:5]) + "/" for n in z.namelist() if len(p := n.split("/")) == 6 and p[1] == args.subject and p[4] in rows}
    if set(prefixes) != set(rows):
        raise ValueError("One or both requested series are not present in the completed archive")
    args.output.mkdir(parents=True, exist_ok=True)
    scans = []
    for image_id in args.images:
        row = rows[image_id]
        converted = convert_series(args.archive, prefixes[image_id], args.subject, image_id, args.output)
        metadata = json.loads(Path(converted["sidecar"]).read_text())
        scans.append({"id": f"local-{image_id}", "subject": args.subject, "modality": "MRI",
                      "date": iso_date(row["Acq Date"]), "visit": row["Visit"], "description": row["Description"],
                      "path": str(Path(converted["nifti"]).resolve()), "space": f"sub-{args.subject}:native:{image_id}",
                      "units": "arbitrary intensity", "registration": "native",
                      "qc": "Converted native head MRI · skull stripping and segmentation not performed",
                      "provenance": "Existing local IDA DICOM series → dcm2niix. No registration, normalization, brain extraction, or quantitative interpretation.",
                      "metadata": {"image_id": image_id, "archive": str(args.archive), "manufacturer": metadata.get("Manufacturer"),
                                   "model": metadata.get("ManufacturersModelName"), "field_t": metadata.get("MagneticFieldStrength"),
                                   "protocol": metadata.get("ProtocolName"), "sidecar": metadata}})
    manifest = args.output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if any(s["id"] not in {s["id"] for s in scans} for s in old.get("scans", [])):
            raise ValueError("Existing manifest contains other scans; choose a separate output folder")
    manifest.write_text(json.dumps({"version": 1, "subjects": [{"id": args.subject, "group": rows[args.images[0]]["Group"],
                         "cohort": rows[args.images[0]]["Group"], "sex": rows[args.images[0]]["Sex"]}], "scans": scans}, indent=2))
    print(f"Prepared {len(scans)} native MRI scans: {manifest}")


if __name__ == "__main__":
    main()
