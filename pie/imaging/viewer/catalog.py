"""Discover finished PIE outputs without modifying any analysis products.

Dates, visits, archive groups and current cohorts are separate concepts. In
particular, repeated series acquired on one date are NOT longitudinal visits.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MODALITIES = ("MRI", "DTI", "SPECT", "PET", "CT", "fMRI")


def rows(path: Path) -> list[dict]:
    if not path.is_file():
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


def token(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


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


class Catalog:
    def __init__(self, repo: Path, ppmi: Path | None = None, manifest: Path | None = None):
        self.repo = repo.resolve()
        self.ppmi = ppmi or repo / "PPMI"
        self.scans: dict[str, Scan] = {}
        self.subjects: dict[str, dict] = {}
        self.warnings: list[str] = []
        self.manifest = manifest
        self.anatomy_previews = {}
        self.discover()

    def discover(self):
        root = self.repo / "Imaging/derived"
        statuses = {r["PATNO"]: r for r in rows(self.ppmi / "_Subject_Characteristics/Participant_Status_08Sep2026.csv")}
        sessions = rows(root / "sessions.csv")
        for row in sessions:
            subject, image_id = row["patno"], row["image_id"]
            fs = root / "fastsurfer" / image_id / "mri"
            image = fs / "orig_nu.mgz"
            if not image.is_file():
                image = fs / "orig.mgz"
            if not image.is_file():
                continue
            status = statuses.get(subject, {})
            group = row.get("ida_group") or row.get("loni_group") or status.get("COHORT_DEFINITION") or "Unknown"
            self.subjects.setdefault(subject, {
                "id": subject, "group": group,
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
                atlas_name="DKT + aseg", atlas_lut=Path(__file__).with_name("atlas.tsv"),
            )
            self.scans[scan.id] = scan
        # The diffusion pipeline retains one selected session per participant.
        # Only attach an acquisition date when selected index rows agree on it.
        index: dict[str, list[dict]] = {}
        for row in rows(root / "dwi/dwi_index.csv"):
            if row.get("selected", "").lower() == "true":
                index.setdefault(row["patno"], []).append(row)
        for subject, selected in index.items():
            d = root / "dwi" / subject
            if subject not in self.subjects or not (d / "fa.nii.gz").is_file() or not (d / "b0.nii.gz").is_file():
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
                atlas_name="DKT + aseg", atlas_lut=Path(__file__).with_name("atlas.tsv"),
            )
            for name, label, units in (("md", "Mean diffusivity", "mm²/s"), ("fw", "Free-water fraction", "fraction"), ("fat", "Tissue fractional anisotropy", "FA (dimensionless)")):
                if (d / f"{name}.nii.gz").is_file():
                    scan.extra.append({"name": label, "path": d / f"{name}.nii.gz", "units": units, "key": name})
            self.scans[scan.id] = scan
        spect_meta = {r["Image Data ID"]: r for r in rows(self.repo / "Imaging/First_Study_SPECT_9_07_2026.csv")}
        for row in rows(root / "datscan_v5/datscan_sbr.csv"):
            if row.get("error") or row["patno"] not in self.subjects:
                continue
            path = self.repo / row["nifti"]
            if not path.is_file():
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
        if self.manifest and self.manifest.is_file():
            self.load_manifest(self.manifest)
        from .anatomy import discover_previews
        self.anatomy_previews = discover_previews(self.repo, self.scans)
        for scan_id, preview in self.anatomy_previews.items():
            self.scans[scan_id].metadata["anatomy_preview"] = preview.public()

    def load_manifest(self, path: Path):
        """Load explicit local files; the HTTP API never accepts arbitrary file paths."""
        doc = json.loads(path.read_text())
        if doc.get("version") != 1:
            raise ValueError("Viewer manifest must use version 1")
        for s in doc.get("subjects", []):
            self.subjects[str(s["id"])] = {"group": "Imported", "cohort": "Imported", **s, "id": str(s["id"])}
        for item in doc.get("scans", []):
            item = dict(item)
            if item["modality"] not in MODALITIES:
                raise ValueError(f"Unknown modality: {item['modality']}")
            item["subject"] = str(item["subject"])
            if item["id"] in self.scans:
                raise ValueError(f"Duplicate scan id: {item['id']}")
            if item.get("date") and iso_date(item["date"]) != item["date"]:
                raise ValueError("Scan dates must be real ISO dates (YYYY-MM-DD), or null")
            item.setdefault("date", None)
            item.setdefault("visit", "Imported acquisition")
            item.setdefault("description", item["modality"])
            item.setdefault("space", f"sub-{item['subject']}:native:{item['id']}")
            for key in ("path", "mask", "atlas", "anatomy", "atlas_lut"):
                if item.get(key):
                    p = (path.parent / item[key]).resolve()
                    if not p.is_file():
                        raise ValueError(f"Missing {key} file for {item['id']}: {p}")
                    item[key] = p
            for extra in item.get("extra", []):
                extra["path"] = (path.parent / extra["path"]).resolve()
                if not extra["path"].is_file():
                    raise ValueError(f"Missing diffusion metric file: {extra['path']}")
            if item.get("kind", "scalar") not in ("scalar", "timeseries", "tracts"):
                raise ValueError("kind must be scalar, timeseries, or tracts; raw SPECT projections must be reconstructed first")
            if item.get("registration", "native") not in ("native", "verified"):
                raise ValueError("registration must be native or verified")
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
