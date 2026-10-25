"""Pinned XCP-D post-processing of fMRIPrep derivatives, and PIE feature rows from its outputs.

XCP-D does denoising, censoring, parcellation, ALFF and ReHo; PIE keeps the PPMI-specific layers: the BIDS
export, provenance, and striatal connectivity recomputed on XCP-D's denoised BOLD. Run one protocol (one TR)
per configuration: ``python -m pie.imaging.xcpd --config settings.json --dry-run``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from .fmri_connectivity import (RESPIRATORY_BAND_BPM, ConnectivityConfig, network_means, roi_timeseries,
                                seed_network_connectivity, striatal_rois)
from .fmriprep import container_runtime, execute_recorded, tree_hashes

XCPD_VERSION = "26.2.0"           # 2026-08-17; first release with --censor-between
# The image bakes its TemplateFlow templates (and its atlases) into /home/xcp_d; PIE binds its own home
# elsewhere, so point TemplateFlow there and nothing is downloaded at run time.
IMAGE_TEMPLATEFLOW = "/home/xcp_d/.cache/templateflow"
# Flags PIE sets itself, so the recorded configuration stays the whole story (compared with "_" as "-").
_RESERVED = {"-w", "--work-dir", "--participant-label", "--mode", "--input-type", "--file-format", "-t",
             "--task-id", "--dummy-scans", "--despike", "-p", "--nuisance-regressors", "--motion-filter-type",
             "--band-stop-min", "--band-stop-max", "--motion-filter-order", "-f", "--fd-thresh",
             "--censor-between", "--min-time", "--output-type", "--disable-bandpass-filter", "--lower-bpf",
             "--upper-bpf", "--bpf-order", "-r", "--head-radius", "--smoothing", "--atlases", "--skip",
             "--min-coverage", "--abcc-qc", "--linc-qc", "-m", "--combine-runs", "--output-layout",
             "--fs-license-file", "--config-file", "--nprocs", "--nthreads", "--n-cpus", "--omp-nthreads", "--mem-mb",
             "--random-seed"}
# BIDS entities XCP-D copies into its QC TSV: identifiers, never features
_ENTITIES = {"sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo", "part", "space", "res", "den",
             "cohort", "desc"}


@dataclass
class XCPDConfig:
    fmriprep_dir: str
    output_dir: str
    work_dir: str
    cache_dir: str
    participants: list[str]
    repetition_time: float
    backend: str = "native"
    executable: str = "xcp_d"
    container_image: Optional[str] = None
    expected_version: Optional[str] = XCPD_VERSION
    nprocs: int = 4
    omp_nthreads: int = 1
    memory_mb: int = 12000
    nuisance_regressors: str = "36P"
    fd_threshold: float = 0.3
    censor_between: int = 4
    min_time: float = 300
    smoothing: float = 6
    atlases: list[str] = field(default_factory=lambda: ["4S456Parcels", "4S156Parcels"])
    min_coverage: float = 0.5
    fs_license_file: Optional[str] = None
    random_seed: int = 42
    extra_args: list[str] = field(default_factory=list)
    motion_filter: Optional[bool] = None     # None: from the TR (notch at TR <= 1.5 s); False: off (no-notch variant)

    def paths(self):
        return {key: Path(getattr(self, key)).expanduser().resolve()
                for key in ("fmriprep_dir", "output_dir", "work_dir", "cache_dir")}

    def validate(self):
        if self.backend not in {"native", "apptainer"}:
            raise ValueError("backend must be native or apptainer")
        if not self.participants or any(not re.fullmatch(r"[A-Za-z0-9]+", p) for p in self.participants):
            raise ValueError("Use explicit alphanumeric participant labels without sub- prefixes")
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("Duplicate participant labels")
        if min(self.nprocs, self.omp_nthreads, self.memory_mb) < 1 or self.omp_nthreads > self.nprocs:
            raise ValueError("Invalid CPU/memory limits")
        if (not 0 < self.repetition_time < 10 or self.fd_threshold <= 0 or self.censor_between < 0 or
                self.min_time < 0 or self.smoothing < 0 or not 0 <= self.min_coverage <= 1 or not self.atlases):
            raise ValueError("Invalid XCP-D settings")
        paths = self.paths()
        description = paths["fmriprep_dir"] / "dataset_description.json"
        if not description.is_file() or json.loads(description.read_text()).get("DatasetType") != "derivative":
            raise ValueError("fmriprep_dir must be an fMRIPrep derivative dataset")
        for p in self.participants:
            sidecars = sorted((paths["fmriprep_dir"] / ("sub-" + p)).rglob("*task-rest*_desc-preproc_bold.json"))
            if not sidecars:
                raise ValueError(f"No fMRIPrep resting BOLD for sub-{p}")
            # One protocol per run: the motion filter depends on the TR (PPMI: 2.4, 2.5 and 1.0 s)
            for sidecar in sidecars:
                tr = json.loads(sidecar.read_text()).get("RepetitionTime")
                if tr is None or not np.isclose(tr, self.repetition_time, rtol=0.01):
                    raise ValueError(f"{sidecar.name}: RepetitionTime {tr} is not {self.repetition_time} s")
        values = list(paths.values())
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                if a == b or a in b.parents or b in a.parents:
                    raise ValueError("fMRIPrep, output, work and cache directories must be separate, nonnested paths")
        if self.backend == "apptainer" and (not self.container_image or not Path(self.container_image).is_file()):
            raise ValueError("Apptainer requires an existing local container image; PIE does not pull implicitly")
        if self.fs_license_file and not Path(self.fs_license_file).is_file():
            raise ValueError("FreeSurfer license file does not exist")
        if any(x.split("=", 1)[0].replace("_", "-") in _RESERVED for x in self.extra_args):
            raise ValueError("Set this in the configuration, not extra_args")
        if self.motion_filter and self.repetition_time > ConnectivityConfig.motion_filter_max_tr:
            raise ValueError("The respiratory motion filter applies only at TR <= 1.5 s")


def build_command(config: XCPDConfig, *, version_only=False):
    """Return argv/environment without writes or shell interpolation.

    Every setting that changes results is explicit, so a mode default changing between releases cannot move
    them: ``linc`` mode on NIfTI, ``36P`` with despiking, 0.01-0.08 Hz band-pass, FD (head radius 50 mm) above
    ``fd_threshold`` censored, retained stretches of ``censor_between`` frames or fewer censored too,
    ``min_time`` seconds required, censored outputs. At TR <= 1.5 s the motion parameters are notch-filtered
    over the respiratory band before FD (``RESPIRATORY_BAND_BPM``, as ``fmri_connectivity.nuisance_design``).
    """
    config.validate()
    prefix, env, (fmriprep, output, work), license_path = container_runtime(
        config, config.paths()["fmriprep_dir"], "xcp_d", templateflow=IMAGE_TEMPLATEFLOW)
    if version_only:
        return prefix + ["--version"], env
    notch = config.repetition_time <= ConnectivityConfig.motion_filter_max_tr if config.motion_filter is None else config.motion_filter
    if notch:
        motion = ["--motion-filter-type", "notch", "--band-stop-min", str(RESPIRATORY_BAND_BPM[0]),
                  "--band-stop-max", str(RESPIRATORY_BAND_BPM[1]), "--motion-filter-order", "4"]
    else:
        motion = ["--motion-filter-type", "none"]
    args = [fmriprep, output, "participant", "--participant-label", *config.participants, "-w", work,
            "--mode", "linc", "--input-type", "fmriprep", "--file-format", "nifti", "--task-id", "rest",
            "--dummy-scans", "auto", "--despike", "y", "--nuisance-regressors", config.nuisance_regressors,
            *motion, "--head-radius", "50", "--fd-thresh", str(config.fd_threshold),
            "--censor-between", str(config.censor_between), "--min-time", str(config.min_time),
            "--output-type", "censored", "--lower-bpf", "0.01", "--upper-bpf", "0.08", "--bpf-order", "2",
            "--smoothing", str(config.smoothing), "--atlases", *config.atlases,
            "--min-coverage", str(config.min_coverage), "--abcc-qc", "n", "--linc-qc", "y", "--combine-runs", "n",
            "--nprocs", str(config.nprocs), "--omp-nthreads", str(config.omp_nthreads),
            "--mem-mb", str(config.memory_mb), "--random-seed", str(config.random_seed), "--notrack"]
    if license_path:
        args += ["--fs-license-file", license_path]
    return prefix + args + config.extra_args, env


def run_xcpd(config: XCPDConfig):
    """Execute/resume XCP-D once per identity: configuration, version, container and the SHA-256 of every
    participant's fMRIPrep file, so a rerun fMRIPrep cannot reuse stale XCP-D outputs. Process success is not QC."""
    command, env = build_command(config)
    root = config.paths()["fmriprep_dir"]
    identity = {"config": asdict(config),
                "fmriprep_hashes": {p: tree_hashes(root / ("sub-" + p)) for p in config.participants}}
    return execute_recorded(config, command, build_command(config, version_only=True)[0], env, identity,
                            tool="XCP-D", identity_file="pie_xcpd_identity.json")


def _one(folder, pattern):
    hits = sorted(Path(folder).rglob(pattern))
    if len(hits) > 1:
        raise ValueError(f"Ambiguous XCP-D outputs for {pattern}: {[h.name for h in hits]}")
    return hits[0] if hits else None


def _atlas(xcpd_dir, atlas):
    """{parcel name: group} and the cortical parcel names from XCP-D's atlas table. The group is the parcel's
    network, else its source atlas (4S: CIT168Subcortical, ThalamusHCP, ...), else its name, as XCP-D's own plots."""
    tables = sorted(p for root in ("atlases", "sourcedata/atlases")
                    for p in (Path(xcpd_dir) / root).glob(f"**/*atlas-{atlas}_*dseg.tsv"))
    if not tables:
        raise ValueError(f"No XCP-D atlas table for {atlas}")
    df = pd.read_table(tables[0])
    df = df[df["index"] > 0].sort_values("index")
    names = df["name"].astype(str)
    group = names
    for column in ("atlas_name", "network_label"):
        if column in df:
            group = df[column].where(df[column].notna(), group).astype(str)
    cortical = set(names[df["network_label"].notna()]) if "network_label" in df else set()
    return dict(zip(names, group)), cortical


def xcpd_edges(xcpd_dir, subject, *, atlas="4S456Parcels", task="rest"):
    """(parcel names, Fisher-z upper-triangle edges in ``np.triu_indices`` order) from XCP-D's Pearson matrix, or
    None when XCP-D wrote none. Parcels below ``--min-coverage`` are NaN, as XCP-D leaves them. Input to ``qc_fc``."""
    path = _one(Path(xcpd_dir) / f"sub-{subject}",
                f"sub-{subject}_*task-{task}_*atlas-{atlas}_*stat-pearsoncorrelation_relmat.tsv")
    if path is None:
        return None
    matrix = pd.read_table(path, index_col="Node")
    r = matrix.to_numpy(float)
    return list(matrix.columns), np.arctanh(np.clip(r[np.triu_indices(len(r), 1)], -1 + 1e-7, 1 - 1e-7))


def xcpd_features(xcpd_dir, subject, *, atlases=("4S456Parcels",), task="rest"):
    """One feature row from XCP-D's outputs for ``sub-<subject>``.

    Columns start ``fmri_`` so ``pie.experiment.prediction`` leaves them unadjusted for intracranial volume.
    ``fmri_xcpd_qc_*``: XCP-D's LINC QC TSV (``mean_fd``, ``mean_fd_post_censoring``, ``num_retained_volumes``,
    ...) plus ``fmri_xcpd_qc_retained_seconds``. Per atlas: ``fmri_xcpd_fc_<atlas>_<A>__<B>`` (mean Fisher z per
    network pair), ``fmri_xcpd_malff_<atlas>_<group>`` (ALFF over its mean across parcels: XCP-D's ALFF is in
    scanner units) and ``fmri_xcpd_reho_<atlas>_<group>`` network means, ``fmri_xcpd_coverage_min_<atlas>`` and
    ``fmri_xcpd_parcels_nan_<atlas>`` (parcels XCP-D blanked for coverage, left out of the means). A run XCP-D did
    not write (``--min-time``, failure) gives ``{"fmri_xcpd_available": False}``, a visible exclusion.
    """
    func = Path(xcpd_dir) / f"sub-{subject}"
    qc = _one(func, f"sub-{subject}_*task-{task}_*desc-linc_qc.tsv")
    bold = _one(func, f"sub-{subject}_*task-{task}_*desc-denoised_bold.nii.gz")
    if qc is None or bold is None:
        return {"fmri_xcpd_available": False}
    table = pd.read_table(qc)
    if len(table) != 1:
        raise ValueError(f"{qc.name}: expected one row")
    row = {"fmri_xcpd_available": True}
    row |= {"fmri_xcpd_qc_" + k: float(v) for k, v in table.iloc[0].items()
            if k not in _ENTITIES and isinstance(v, (int, float, np.number)) and not isinstance(v, bool)}
    tr = json.loads(Path(str(bold).replace(".nii.gz", ".json")).read_text())["RepetitionTime"]
    row["fmri_xcpd_qc_retained_seconds"] = row["fmri_xcpd_qc_num_retained_volumes"] * tr
    for atlas in atlases:
        groups, _ = _atlas(xcpd_dir, atlas)
        found = xcpd_edges(xcpd_dir, subject, atlas=atlas, task=task)
        if found is None:
            raise ValueError(f"XCP-D wrote BOLD but no {atlas} matrix for sub-{subject}")
        names, edges = found
        row |= {f"fmri_xcpd_fc_{atlas}_{k}": v for k, v in network_means(edges, [groups[n] for n in names]).items()}
        coverage = pd.read_table(_one(func, f"sub-{subject}_*task-{task}_*atlas-{atlas}_*stat-coverage_bold.tsv"),
                                 index_col="Node")["coverage"]
        row[f"fmri_xcpd_coverage_min_{atlas}"] = float(coverage.min())
        series = pd.read_table(_one(func, f"sub-{subject}_*task-{task}_*atlas-{atlas}_*stat-mean_timeseries.tsv"))
        row[f"fmri_xcpd_parcels_nan_{atlas}"] = int(series.isna().all().sum())
        for stat, prefix in (("alff", "malff"), ("reho", "reho")):
            path = _one(func, f"sub-{subject}_*task-{task}_*atlas-{atlas}_*stat-{stat}_bold.tsv")
            if path is None:      # no ALFF without band-pass filtering
                continue
            values = pd.read_table(path).iloc[0].astype(float)
            if stat == "alff":
                values = values / values.mean()
            means = values.groupby([groups[n] for n in values.index]).mean()
            row |= {f"fmri_xcpd_{prefix}_{atlas}_{g}": float(v) for g, v in means.items()}
    return row


def xcpd_striatal_features(xcpd_dir, subject, *, atlas="4S456Parcels", task="rest", putamen_split_y=0.0,
                           min_voxels=5, frames=None):
    """PIE's striatal seed connectivity on XCP-D's denoised, censored BOLD (MNI152NLin2009cAsym): CIT168 caudate,
    putamen (anterior and posterior at ``putamen_split_y``; None keeps it whole) and accumbens per hemisphere,
    against the cortical networks of ``atlas`` from XCP-D's parcel time series of the same BOLD. Keyed
    ``fmri_xcpd_striatal_<seed>__<network>``; ``{}`` when XCP-D wrote no BOLD. XCP-D's denoising is the nuisance model:
    nothing is regressed again. Holds the BOLD in memory (about 2.6 GB for 600 frames at 2 mm). ``frames`` (a slice
    of the retained frames) gives the split-half features of the fMRI validation."""
    func = Path(xcpd_dir) / f"sub-{subject}"
    bold = _one(func, f"sub-{subject}_*task-{task}_*space-MNI152NLin2009cAsym_*desc-denoised_bold.nii.gz")
    timeseries = _one(func, f"sub-{subject}_*task-{task}_*atlas-{atlas}_*stat-mean_timeseries.tsv")
    if bold is None or timeseries is None:
        return {}
    img = nib.load(bold)
    parcels = pd.read_table(timeseries)
    if len(parcels) != img.shape[3]:
        raise ValueError("Denoised BOLD and parcel time series differ in length: XCP-D must write censored outputs")
    if frames is not None:
        img, parcels = img.slicer[..., frames], parcels.iloc[frames]
    groups, cortical = _atlas(xcpd_dir, atlas)
    columns = [c for c in parcels.columns if c in cortical]
    rois, names = striatal_rois(img, putamen_split_y=putamen_split_y)
    series, labels, _ = roi_timeseries(img, rois, min_voxels=min_voxels)
    z = seed_network_connectivity(series, [names[k] for k in labels], parcels[columns].to_numpy(float),
                                  [groups[c] for c in columns])
    return {"fmri_xcpd_striatal_" + k: v for k, v in z.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON fields matching XCPDConfig")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and print argv without running")
    args = parser.parse_args(argv)
    config = XCPDConfig(**json.loads(args.config.read_text()))
    if args.dry_run:
        print(json.dumps({"argv": build_command(config)[0]}, indent=2))
    else:
        print(json.dumps(run_xcpd(config), indent=2))


if __name__ == "__main__":
    main()
