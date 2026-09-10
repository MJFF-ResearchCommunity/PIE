"""Evidence-backed acquisition shortlist from locally downloaded PPMI tables.

An acquisition form establishes a candidate, not IDA image availability.
Nothing here logs in to IDA or downloads restricted imaging automatically.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .catalog import Catalog, MODALITIES, rows

ARCHIVE_GROUPS = ["Control", "PD", "SWEDD", "Prodromal", "GenCohort PD", "GenCohort Unaff", "GenReg PD", "GenReg Unaff", "AV133", "Volunteer", "Phantom"]


def build_plan(repo: Path, ppmi: Path, output: Path):
    catalog = Catalog(repo, ppmi)
    public = catalog.public()
    by_subject = {s["id"]: s for s in public["subjects"]}
    status = {r["PATNO"]: r for r in rows(ppmi / "_Subject_Characteristics/Participant_Status_08Sep2026.csv")}
    imaging = ppmi / "Imaging"
    candidates = defaultdict(dict)

    def add(modality, subject, date, visit, source, evidence, tracer=""):
        cohort = status.get(subject, {}).get("COHORT_DEFINITION", "Unknown")
        entry = candidates[(modality, cohort)].setdefault(subject, {
            "modality": modality, "cohort": cohort, "subject": subject, "dates": set(), "visits": set(),
            "sources": set(), "evidence": set(), "tracers": set(),
        })
        for key, val in (("dates", date), ("visits", visit), ("sources", source), ("evidence", evidence), ("tracers", tracer)):
            if val:
                entry[key].add(val)

    for name in ("MRI_Acquisition_Metadata_18Mar2025.csv",):
        for r in rows(imaging / name):
            for modality, column in (("MRI", None), ("DTI", "MRI_SEQ_DTI"), ("fMRI", "MRI_SEQ_RS")):
                if column and r.get(column, "").lower() != "yes":
                    continue
                if not r.get("MRI_SCAN_DATE"):
                    continue
                add(modality, r["PATNO"], r["MRI_SCAN_DATE"], r.get("EVENT_ID"), name, f"{column}=Yes" if column else "MRI_SCAN_DATE recorded; review image QC")
    name = "DaTScan_Acquisition_Metadata_18Mar2025.csv"
    for r in rows(imaging / name):
        if r.get("DATSCAN_IMAGE_ACCEPTABLE", "").lower() == "no":
            continue
        if r.get("DATSCAN_DATE"):
            add("SPECT", r["PATNO"], r["DATSCAN_DATE"], r.get("EVENT_ID"), name,
                "DaTscan acquisition recorded; request reconstructed image and review QC",
                r.get("DATSCAN_LIGAND") or "DaTscan (verify ligand in acquisition metadata)")
    name = "DaTscan_Imaging_18Mar2025.csv"
    for r in rows(imaging / name):
        # Annotated PPMI code list: 1 = completed at this visit; 0 = not
        # completed; 2 = a pre-consent scan, not a new acquisition at this visit.
        if r.get("DATSCAN") == "1":
            tracer = "Beta-CIT" if r.get("SCNINJCT") == "2" else {"1": "DaTscan", "2": "TRODAT"}.get(r.get("DATSCANTRC"), "Verify tracer in metadata")
            add("SPECT", r["PATNO"], r.get("INFODT"), r.get("EVENT_ID"), name,
                "DATSCAN=1 (completed at this visit); assessment month is a search hint; confirm reconstruction and QC", tracer)
    name = "PET_Acquisition_Metadata_08Sep2026.csv"
    for r in rows(imaging / name):
        if r.get("PET_IMAGE_ACCEPTABLE", "").lower() == "no" or r.get("PET_PASS_QC", "").lower() == "no":
            continue
        if r.get("PET_SCAN_DATE"):
            add("PET", r["PATNO"], r["PET_SCAN_DATE"], r.get("EVENT_ID"), name, "PET acquisition recorded; not explicitly QC-rejected", r.get("PET_LIGAND", ""))
    for name in ("CT_Scan_18Mar2025.csv", "Safety_Head_CT_Scan_08Sep2026.csv"):
        for r in rows(imaging / name):
            if r.get("CTSCAN") == "1":
                add("CT", r["PATNO"], r.get("CTDT"), r.get("EVENT_ID"), name, "CTSCAN=1 (performed); verify IDA image access and series purpose")
    for subject in public["subjects"]:
        for s in subject["scans"]:
            add(s["modality"], subject["id"], s["date"], s["visit"], "Local PIE outputs", "Image file present", s["tracer"] or "")

    shortlist = []
    coverage = []
    cohorts = sorted({v.get("COHORT_DEFINITION", "Unknown") for v in status.values()})
    for cohort in cohorts:
        for modality in MODALITIES:
            options = list(candidates[(modality, cohort)].values())
            options.sort(key=lambda x: (
                -int("MRI" in by_subject.get(x["subject"], {}).get("modalities", [])),
                -len(by_subject.get(x["subject"], {}).get("modalities", [])),
                -len(x["dates"]), x["subject"],
            ))
            local = sum(modality in s["modalities"] and s["cohort"] == cohort for s in public["subjects"])
            coverage.append({"cohort": cohort, "modality": modality, "local_subjects": local, "candidate_subjects": len(options)})
            for rank, option in enumerate(options[:2], 1):
                shortlist.append({**{k: sorted(v) if isinstance(v, set) else v for k, v in option.items()}, "rank": rank,
                                  "local_modalities": by_subject.get(option["subject"], {}).get("modalities", []),
                                  "availability": "Candidate only; confirm in IDA search"})

    legacy = []
    for group in ARCHIVE_GROUPS:
        matching = [s for s in public["subjects"] if s["group"] == group]
        matching.sort(key=lambda s: (-len(s["modalities"]), s["id"]))
        legacy.append({"archive_group": group, "local_subjects": len(matching), "suggested_subjects": [s["id"] for s in matching[:2]],
                       "note": "Archive label; preserve separately from current cohort" if matching else "Not established by this local inventory; run an unrestricted-date IDA query for this group"})
    plan = {"generated_from": "Local PPMI tables and finished PIE outputs", "coverage": coverage, "shortlist": shortlist, "archive_groups": legacy,
            "notes": ["Acquisition dates from clinical tables often have month precision. These are search hints, not exact acquisition timestamps.",
                      "CTSCAN=0 means not performed and is excluded.",
                      "Research Group checkboxes reflect archive labels, not necessarily current clinical cohorts.",
                      "A complete modality × cohort matrix is not guaranteed; retain explicit unavailable cells.",
                      "Phantom is scanner QA, not a patient cohort. AV133 is an imaging/archive category, not a diagnosis."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    flat = [{k: "; ".join(v) if isinstance(v, list) else v for k, v in r.items()} for r in shortlist]
    if flat:
        with (output / "download_candidates.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    for modality in MODALITIES:
        ids = sorted({r["subject"] for r in shortlist if r["modality"] == modality})
        (output / f"{modality}_subject_ids.txt").write_text(",".join(ids) + "\n")
    report = ["# PPMI viewer sample download plan", "", "Generated from local acquisition tables; confirm actual images in IDA. Clinical month dates are search hints.", "",
              "| Current cohort | Modality | Local participants | Acquisition candidates |", "|---|---|---:|---:|"]
    report += [f"| {r['cohort']} | {r['modality']} | {r['local_subjects']} | {r['candidate_subjects']} |" for r in coverage]
    report += ["", "## Suggested participants", "", "| Modality | Cohort | PATNO | Visits / months | Evidence |", "|---|---|---|---|---|"]
    report += [f"| {r['modality']} | {r['cohort']} | {r['subject']} | {', '.join(r['dates'][:6])} | {'; '.join(r['evidence'])} |" for r in shortlist]
    report += ["", "## Archive research groups", ""]
    report += [f"- {r['archive_group']}: {', '.join(r['suggested_subjects']) or 'search IDA; no established local sample'}. {r['note']}." for r in legacy]
    report += ["", "## Download priorities", "", "1. Use MRI_subject_ids.txt / DTI_subject_ids.txt / fMRI_subject_ids.txt etc. in Subject ID. Select each modality separately; don't require all modalities on the same participant.",
               "2. Start with two participants per available cohort, T1w 3D anatomy plus matching DTI / resting-state BOLD / reconstructed SPECT or PET, and two genuinely different visits (baseline and 12 or 24 months where available). Screening SPECT can precede baseline MRI; preserve both dates.",
               "3. MRI: T1w MPRAGE/BRAVO/FSPGR, preferably 3D and around 1 mm isotropic; include T2/FLAIR if wanted. DTI: all diffusion directions, bvals/bvecs, phase-encoding/readout sidecars and reverse-phase b0. fMRI: full 4D resting BOLD, TR, fieldmaps/reverse-phase EPI, and anatomical reference.",
               "4. SPECT: prefer the PPMI processed reconstructed, corrected 3D volume and processing metadata. Original NM can be projection angles rather than anatomical z slices. PET: keep tracer, units, injection time/dose, acquisition timing and reconstruction metadata; do not mix tracer intensity scales.",
               "5. CT: use the performed-CT candidates only, then confirm a brain volume is downloadable. Distinguish diagnostic head CT from low-dose attenuation-correction CT. Don't interpret the CT checkbox as evidence every cohort has CT.",
               "6. Export the IDA collection CSV and Advanced Download metadata with every collection. Keep actual acquisition date, image ID, participant ID, visit, group, protocol, voxel spacing and image type.",
               "7. For missing cohort/modality cells or Volunteer/Phantom/AV133 archive categories, query IDA without the Subject ID restriction. If no matching data exist, record unavailable rather than substituting a different cohort.", ""]
    (output / "DOWNLOAD_PLAN.md").write_text("\n".join(report))
    print(f"Wrote {len(shortlist)} evidence-backed candidate rows to {output}")
    print("Local modality counts:", dict(Counter(s.modality for s in catalog.scans.values())))
