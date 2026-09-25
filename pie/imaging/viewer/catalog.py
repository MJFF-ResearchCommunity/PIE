"""Discover finished PIE outputs without modifying any analysis products.

Dates, visits, archive groups and current cohorts are separate concepts. In
particular, repeated series acquired on one date are NOT longitudinal visits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from pathlib import Path

MODALITIES = ("MRI", "DTI", "SPECT", "PET", "CT", "fMRI")
DKT_ATLAS = "DKT + aseg"
DKT_LUT = Path(__file__).with_name("atlas.tsv")
# PPMI re-releases tables with the download date in the name: 08Sep2026, 9_07_2026, ...
TABLE_DATE_FORMATS = ("%d%b%Y", "%m_%d_%Y", "%Y-%m-%d", "%Y%m%d")


def rows(path: Path | None) -> list[dict]:
    if not path or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def iso_date(value: str) -> str | None:
    """Keep only actual day-precision dates; masked and month-only dates are not days."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            date = datetime.strptime(str(value)[:10], fmt).date()
            return date.isoformat() if 1900 <= date.year <= 2100 else None
        except ValueError:
            pass
    return None


def table_date(stamp: str) -> date | None:
    for fmt in TABLE_DATE_FORMATS:
        try:
            return datetime.strptime(stamp, fmt).date()
        except ValueError:
            pass
    # a versioned table carries its own date before the release date: iu_genetic_consensus_20251025_08Sep2026
    parts = [table_date(p) for p in stamp.split("_")] if "_" in stamp else [None]
    return parts[-1] if all(parts) else None


def latest_table(directory: Path, stem: str) -> Path | None:
    """Newest `<stem>_<release date>.csv` (or an undated `<stem>.csv`) in a directory.

    Only a parseable date may follow the stem, so `CT_Scan` never picks up an
    unrelated `CT_Scan_notes.csv`. Equal dates fall back to modification time.
    """
    pattern = re.compile(re.escape(stem) + r"(?:_(.+))?\.csv")
    found = []
    for path in directory.glob(f"{stem}*.csv") if directory.is_dir() else []:
        match = pattern.fullmatch(path.name)
        stamp = match and (table_date(match[1]) if match[1] else date.min)
        if stamp:
            found.append((stamp, path.stat().st_mtime, path))
    return max(found)[2] if found else None


def token(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def source_fingerprint(archive, members) -> str:
    """Identity of a converted series: its archive and (name, size, CRC) members."""
    return hashlib.sha256(json.dumps([str(archive), sorted(members)], sort_keys=True).encode()).hexdigest()


def read_manifest(path: Path) -> dict:
    """An existing version-1 manifest document, or a new empty one."""
    if not path.exists():
        return {"version": 1, "subjects": [], "scans": []}
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(doc, dict) or doc.get("version") != 1:
        raise ValueError(f'{path}: a viewer manifest must be a JSON object with "version": 1')
    doc.setdefault("subjects", [])
    doc.setdefault("scans", [])
    if not isinstance(doc["subjects"], list) or not isinstance(doc["scans"], list):
        raise ValueError(f'{path}: "subjects" and "scans" must be lists')
    return doc


def same_source(old: dict, new: dict) -> bool:
    a, b = old.get("metadata", {}), new.get("metadata", {})
    if a.get("source_fingerprint") and b.get("source_fingerprint"):
        return a["source_fingerprint"] == b["source_fingerprint"]
    # Entries written before fingerprints were recorded: same archive and image ID.
    return (a.get("archive"), a.get("image_id")) == (b.get("archive"), b.get("image_id"))


def merge_manifest(path: Path, subjects: list[dict], scans: list[dict]) -> dict:
    """Add or refresh scans by ID and keep every other entry, so prepare scripts
    can share one collection manifest in any order. Reusing an ID for a
    different source is refused before anything is written."""
    doc = read_manifest(path)
    known = {str(s.get("id")) for s in doc["subjects"]}
    for subject in subjects:
        if str(subject["id"]) not in known:
            doc["subjects"].append(subject)
            known.add(str(subject["id"]))
    by_id = {s.get("id"): s for s in doc["scans"]}
    for scan in scans:
        old = by_id.get(scan["id"])
        if old and not same_source(old, scan):
            raise ValueError(f"{scan['id']}: {path} already has this ID from a different source; no manifest changes saved")
        by_id[scan["id"]] = scan
    doc["scans"] = list(by_id.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(doc, indent=2))
    temporary.replace(path)
    return doc


@dataclass
class Scan:
    id: str
    subject: str
    modality: str
    date: str | None
    visit: str
    description: str
    path: Path
    space: str
    kind: str = "scalar"
    units: str = "arbitrary intensity"
    atlas: Path | None = None
    mask: Path | None = None
    anatomy: Path | None = None
    reference_id: str | None = None
    registration: str = "native"
    provenance: str = ""
    qc: str = "Not reviewed in viewer"
    tracer: str | None = None
    metadata: dict = field(default_factory=dict)
    extra: list[dict] = field(default_factory=list)
    atlas_name: str = "Unspecified label atlas"
    atlas_lut: Path | None = None

    def public(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "subject", "modality", "date", "visit", "description", "space",
            "kind", "units", "reference_id", "registration", "provenance", "qc",
            "tracer", "metadata", "atlas_name",
        )} | {"has_atlas": bool(self.atlas), "has_anatomy": bool(self.anatomy)}


SCAN_FIELDS = {f.name for f in fields(Scan)}
MANIFEST_REQUIRED = ("id", "subject", "modality", "path")
EXTRA_FIELDS = {"name", "path", "units", "key"}


class Catalog:
    def __init__(self, repo: Path, ppmi: Path | None = None, manifest: Path | None = None):
        self.repo = repo.resolve()
        self.ppmi = ppmi or repo / "PPMI"
        self.scans: dict[str, Scan] = {}
        self.subjects: dict[str, dict] = {}
        self.statuses: dict[str, dict] = {}
        self.warnings: list[str] = []
        self.manifest = manifest
        self.anatomy_previews = {}
        self.discover()

    def discover(self):
        status_dir = self.ppmi / "_Subject_Characteristics"
        self.statuses = {r.get("PATNO"): r for r in rows(latest_table(status_dir, "Participant_Status"))}
        root = self.repo / "Imaging/derived"
        if root.is_dir():
            self.discover_pie(root, status_dir)
        else:
            self.warnings.append(f"No PIE imaging outputs at {root}; only manifest scans are indexed.")
        if self.manifest:
            self.load_manifest(self.manifest)
        from .anatomy import discover_previews
        self.anatomy_previews = discover_previews(self.repo, self.scans)
        for scan_id, preview in self.anatomy_previews.items():
            self.scans[scan_id].metadata["anatomy_preview"] = preview.public()

    def discover_pie(self, root: Path, status_dir: Path):
        """Finished MRI/DTI/SPECT products. Gaps become warnings, never guesses."""
        sessions_path = root / "sessions.csv"
        if not sessions_path.is_file():
            self.warnings.append(f"No {sessions_path}; no PIE MRI is indexed (DTI and SPECT also need an indexed MRI).")
        sessions = rows(sessions_path)
        if sessions and not self.statuses:
            self.warnings.append(f"No Participant_Status_<date>.csv in {status_dir}; current cohort falls back to the archive group.")
        no_image = 0
        for row in sessions:
            subject, image_id = row["patno"], row["image_id"]
            fs = root / "fastsurfer" / image_id / "mri"
            image = fs / "orig_nu.mgz"
            if not image.is_file():
                image = fs / "orig.mgz"
            if not image.is_file():
                no_image += 1
                continue
            status = self.statuses.get(subject, {})
            group = row.get("ida_group") or row.get("loni_group") or status.get("COHORT_DEFINITION") or "Unknown"
            self.subjects.setdefault(subject, {
                "id": subject, "collection": "PPMI", "group": group,
                "cohort": status.get("COHORT_DEFINITION", group),
                "sex": row.get("loni_sex") or None,
                "age_at_scan": row.get("ida_age") or row.get("loni_age") or None,
            })
            scan = Scan(
                id=f"mri-{image_id}", subject=subject, modality="MRI",
                date=iso_date(row.get("ida_date", "")) or iso_date(row.get("loni_acq_date", "")) or iso_date(row.get("session_date", "")),
                visit=row.get("ida_visit") or row.get("EVENT_ID") or "Unknown visit",
                description=row.get("ida_desc") or row.get("loni_desc") or row["series_desc"].replace("_", " "),
                path=image, space=f"sub-{subject}:T1:{image_id}",
                mask=fs / "mask.mgz" if (fs / "mask.mgz").is_file() else None,
                atlas=fs / "aparc.DKTatlas+aseg.deep.mgz" if (fs / "aparc.DKTatlas+aseg.deep.mgz").is_file() else None,
                provenance="PPMI T1w MRI → PIE FastSurfer conformation / bias correction. Atlas: participant-specific DKT + aseg segmentation.",
                metadata={"image_id": image_id, "manufacturer": row.get("ida_manufacturer"), "model": row.get("ida_model"), "field_t": row.get("ida_field")},
                atlas_name=DKT_ATLAS, atlas_lut=DKT_LUT,
            )
            self.scans[scan.id] = scan
        if no_image:
            self.warnings.append(f"Skipped {no_image} sessions.csv row(s) without a finished FastSurfer orig_nu.mgz/orig.mgz.")
        # The diffusion pipeline retains one selected session per participant.
        # Only attach an acquisition date when selected index rows agree on it.
        index_path = root / "dwi/dwi_index.csv"
        if not index_path.is_file():
            self.warnings.append(f"No {index_path}; diffusion maps are not indexed.")
        index: dict[str, list[dict]] = {}
        for row in rows(index_path):
            if row.get("selected", "").lower() == "true":
                index.setdefault(row["patno"], []).append(row)
        skipped = Counter()
        for subject, selected in index.items():
            d = root / "dwi" / subject
            if subject not in self.subjects:
                skipped["no indexed MRI"] += 1
                continue
            if not (d / "fa.nii.gz").is_file() or not (d / "b0.nii.gz").is_file():
                skipped["fa.nii.gz or b0.nii.gz missing"] += 1
                continue
            dates = {iso_date(r["date"]) for r in selected}
            date = next(iter(dates)) if len(dates) == 1 else None
            scan = Scan(
                id=f"dti-{subject}", subject=subject, modality="DTI", date=date, visit="Diffusion acquisition",
                description="Fractional anisotropy · native diffusion space", path=d / "fa.nii.gz",
                anatomy=d / "b0.nii.gz", atlas=d / "aseg_dwi.nii.gz" if (d / "aseg_dwi.nii.gz").is_file() else None,
                space=f"sub-{subject}:DWI", units="FA (dimensionless)",
                provenance="PIE tensor fit. FA and mean b0 share the native diffusion grid; labels were resampled by the PIE pipeline. No implicit T1 registration.",
                metadata={"image_ids": sorted({r["image_id"] for r in selected}), "date_note": "Selected DWI index" if date else "Acquisition date ambiguous in legacy pipeline index"},
                atlas_name=DKT_ATLAS, atlas_lut=DKT_LUT,
            )
            for name, label, units in (("md", "Mean diffusivity", "mm²/s"), ("fw", "Free-water fraction", "fraction"), ("fat", "Tissue fractional anisotropy", "FA (dimensionless)")):
                if (d / f"{name}.nii.gz").is_file():
                    scan.extra.append({"name": label, "path": d / f"{name}.nii.gz", "units": units, "key": name})
            self.scans[scan.id] = scan
        for reason, n in skipped.items():
            self.warnings.append(f"Skipped {n} selected diffusion participant(s): {reason}.")
        spect_path = root / "datscan_v5/datscan_sbr.csv"
        if not spect_path.is_file():
            self.warnings.append(f"No {spect_path}; reconstructed SPECT is not indexed.")
        spect_rows = rows(spect_path)
        dates_path = latest_table(self.repo / "Imaging", "First_Study_SPECT")
        if spect_rows and not dates_path:
            self.warnings.append(f"No First_Study_SPECT_<date>.csv in {self.repo / 'Imaging'}; SPECT dates and visits are unknown.")
        spect_meta = {r["Image Data ID"]: r for r in rows(dates_path)}
        skipped = Counter()
        for row in spect_rows:
            if row.get("error"):
                skipped["reconstruction error recorded"] += 1
                continue
            if row["patno"] not in self.subjects:
                skipped["no indexed MRI"] += 1
                continue
            path = self.repo / row["nifti"]
            if not path.is_file():
                skipped["NIfTI missing"] += 1
                continue
            subject, image_id = row["patno"], row["image_id"]
            meta = spect_meta.get(image_id, {})
            scan = Scan(
                id=f"spect-{image_id}", subject=subject, modality="SPECT", date=iso_date(meta.get("Acq Date", "")),
                visit=meta.get("Visit", "Unknown visit"), description="DaTscan · reconstructed emission volume",
                path=path, space=f"sub-{subject}:SPECT:{image_id}", units="reconstruction intensity (not SBR)",
                tracer="123I-ioflupane",
                provenance="Local PIE filtered back-projection reconstruction; no attenuation correction. Stored intensity is not SBR, SUV, or a voxelwise disease probability. Legacy registration does not identify its reference image, so cross-modality fusion is disabled.",
                qc="Exploratory reconstruction · review required",
                metadata={"image_id": image_id, "manufacturer": row.get("hdr_manufacturer"), "model": row.get("hdr_model"), "registration_metric": row.get("reg_metric")},
            )
            self.scans[scan.id] = scan
        for reason, n in skipped.items():
            self.warnings.append(f"Skipped {n} SPECT row(s): {reason}.")

    def load_manifest(self, path: Path):
        """Load explicit local files; the HTTP API never accepts arbitrary file paths."""
        if not path.is_file():
            raise ValueError(f"Viewer manifest not found: {path}")
        doc = read_manifest(path)
        for n, s in enumerate(doc["subjects"], 1):
            if not isinstance(s, dict) or s.get("id") in (None, ""):
                raise ValueError(f'{path}: subject #{n} needs an "id"')
            self.subjects[str(s["id"])] = {"group": "Imported", "cohort": "Imported", **s, "id": str(s["id"])}
        for n, item in enumerate(doc["scans"], 1):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: scan #{n} must be a JSON object")
            item = dict(item)
            where = f"{path}: scan {item.get('id') or '#' + str(n)}"
            missing = [k for k in MANIFEST_REQUIRED if item.get(k) in (None, "")]
            if missing:
                raise ValueError(f"{where}: missing required field(s): {', '.join(missing)}")
            unknown = sorted(set(item) - SCAN_FIELDS)
            if unknown:
                raise ValueError(f"{where}: unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(sorted(SCAN_FIELDS))}")
            if item["modality"] not in MODALITIES:
                raise ValueError(f"{where}: unknown modality {item['modality']!r}; use one of {', '.join(MODALITIES)}")
            if not isinstance(item.get("metadata", {}), dict) or not isinstance(item.get("extra", []), list):
                raise ValueError(f"{where}: metadata must be an object and extra a list")
            for extra in item.get("extra", []):
                if not isinstance(extra, dict) or set(extra) != EXTRA_FIELDS:
                    raise ValueError(f"{where}: each extra entry needs exactly {', '.join(sorted(EXTRA_FIELDS))}")
            item["subject"] = str(item["subject"])
            if item["id"] in self.scans:
                raise ValueError(f"{where}: duplicate scan id")
            if item.get("date") and iso_date(item["date"]) != item["date"]:
                raise ValueError(f"{where}: scan dates must be real ISO dates (YYYY-MM-DD), or null")
            item.setdefault("date", None)
            item.setdefault("visit", "Imported acquisition")
            item.setdefault("description", item["modality"])
            item.setdefault("space", f"sub-{item['subject']}:native:{item['id']}")
            for key in ("path", "mask", "atlas", "anatomy", "atlas_lut"):
                if item.get(key):
                    p = (path.parent / item[key]).resolve()
                    if not p.is_file():
                        raise ValueError(f"{where}: missing {key} file {p}")
                    item[key] = p
            item["extra"] = [{**extra, "path": (path.parent / extra["path"]).resolve()} for extra in item.get("extra", [])]
            for extra in item["extra"]:
                if not extra["path"].is_file():
                    raise ValueError(f"{where}: missing extra map file {extra['path']}")
            # A FastSurfer DKT + aseg segmentation uses the bundled lookup table.
            if item.get("atlas_name") == DKT_ATLAS and not item.get("atlas_lut"):
                item["atlas_lut"] = DKT_LUT
            if item.get("kind", "scalar") not in ("scalar", "timeseries", "tracts"):
                raise ValueError("kind must be scalar, timeseries, or tracts; raw SPECT projections must be reconstructed first")
            if item.get("registration", "native") not in ("native", "verified"):
                raise ValueError(f"{where}: registration must be native or verified")
            self.scans[item["id"]] = Scan(**item)
            self.subjects.setdefault(item["subject"], {"id": item["subject"], "group": "Imported", "cohort": "Imported"})
        for scan in self.scans.values():
            if scan.registration == "verified":
                reference = self.scans.get(scan.reference_id)
                if not reference or reference.id == scan.id or reference.subject != scan.subject or reference.space != scan.space:
                    raise ValueError(f"{scan.id}: verified registration requires a same-subject reference_id in the same explicit space")

    def public(self):
        subjects = []
        for subject in self.subjects.values():
            scans = [s.public() for s in self.scans.values() if s.subject == subject["id"]]
            if not scans:
                continue
            scans.sort(key=lambda s: (s["date"] or "9999", MODALITIES.index(s["modality"]), s["id"]))
            subjects.append({**subject, "scans": scans, "dates": sorted({s["date"] for s in scans if s["date"]}), "modalities": sorted({s["modality"] for s in scans})})
        # Put a participant with MRI, DTI, and SPECT first, then stable numeric ID.
        subjects.sort(key=lambda p: (-len(p["modalities"]), int(p["id"]) if p["id"].isdigit() else 10**12, p["id"]))
        return {"subjects": subjects, "modalities": MODALITIES, "warnings": self.warnings,
                "source": "Local PIE imaging outputs", "scan_count": len(self.scans)}
