"""Configurable fMRIPrep backend for PIE (native install or Apptainer).

PIE orchestrates the maintained fMRIPrep executable and records its derivatives;
it does not vendor fMRIPrep or assume a particular researcher's filesystem.
Run ``python -m pie.imaging.fmriprep --config settings.json --dry-run``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Optional


@dataclass
class FMRIPrepConfig:
    bids_dir: str
    output_dir: str
    work_dir: str
    cache_dir: str
    participants: list[str]
    backend: str = "native"
    executable: str = "fmriprep"
    container_image: Optional[str] = None
    expected_version: Optional[str] = None
    nprocs: int = 4
    omp_nthreads: int = 1
    memory_mb: int = 12000
    output_spaces: list[str] = field(default_factory=lambda: ["MNI152NLin2009cAsym:res-2", "T1w"])
    surface_reconstruction: bool = False
    fs_license_file: Optional[str] = None
    random_seed: int = 42
    extra_args: list[str] = field(default_factory=list)
    derivatives: dict[str, str] = field(default_factory=dict)

    def paths(self):
        return {key: Path(getattr(self, key)).expanduser().resolve()
                for key in ("bids_dir", "output_dir", "work_dir", "cache_dir")}

    def validate(self):
        if self.backend not in {"native", "apptainer"}:
            raise ValueError("backend must be native or apptainer")
        if not self.participants or any(not re.fullmatch(r"[A-Za-z0-9]+", p) for p in self.participants):
            raise ValueError("Use explicit alphanumeric participant labels without sub- prefixes")
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("Duplicate participant labels")
        if min(self.nprocs, self.omp_nthreads, self.memory_mb) < 1 or self.omp_nthreads > self.nprocs:
            raise ValueError("Invalid CPU/memory limits")
        paths = self.paths()
        if not (paths["bids_dir"] / "dataset_description.json").is_file():
            raise ValueError("BIDS dataset_description.json is required; fMRIPrep will validate the dataset")
        for p in self.participants:
            if not (paths["bids_dir"] / ("sub-" + p)).is_dir():
                raise ValueError(f"Participant sub-{p} is absent from the BIDS input")
        values = list(paths.values())
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                if a == b or a in b.parents or b in a.parents:
                    raise ValueError("BIDS, output, work and cache directories must be separate, nonnested paths")
        if self.backend == "apptainer" and (not self.container_image or not Path(self.container_image).is_file()):
            raise ValueError("Apptainer requires an existing local container image; PIE does not pull implicitly")
        if self.fs_license_file and not Path(self.fs_license_file).is_file():
            raise ValueError("FreeSurfer license file does not exist")
        if self.surface_reconstruction and not self.fs_license_file:
            raise ValueError("Surface reconstruction requires a configured FreeSurfer license")
        # These switches would invalidate PIE's declared path/participant contract.
        reserved = {"-w", "--work-dir", "--work_dir", "--participant-label", "--participant_label",
                    "--config-file", "--config_file", "--fs-license-file", "--fs_license_file"}
        if any(x.split("=", 1)[0] in reserved for x in self.extra_args):
            raise ValueError("Set work/participant/license fields in the configuration, not extra_args")
        if self.derivatives and any(x.split('=', 1)[0] in {'-d', '--derivatives'} for x in self.extra_args):
            raise ValueError('Use derivatives configuration or extra_args, not both')
        for name, folder in self.derivatives.items():
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
                raise ValueError('Invalid derivative package name')
            path = Path(folder).expanduser().resolve(strict=True)
            if not (path / 'dataset_description.json').is_file():
                raise ValueError('Derivative dataset_description.json is required')
            # Raw input and a reused derivative must be independent datasets:
            # equal or nested paths let one be indexed and hashed as the other.
            for key in ('bids_dir', 'output_dir', 'work_dir', 'cache_dir'):
                other = paths[key]
                if path == other or path in other.parents or other in path.parents:
                    raise ValueError('Reused derivatives must be separate from the BIDS input '
                                     'and writable output/work/cache')


def build_command(config: FMRIPrepConfig, *, version_only=False):
    """Return argv/environment without writes or shell interpolation."""
    config.validate()
    p = config.paths()
    env = dict(os.environ, OMP_NUM_THREADS=str(config.omp_nthreads),
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=str(config.omp_nthreads),
               TMPDIR=str(p["work_dir"] / "tmp"),
               TEMPLATEFLOW_HOME=str(p["cache_dir"] / "templateflow"),
               XDG_CACHE_HOME=str(p["cache_dir"] / "xdg"))
    if config.backend == "native":
        prefix = [config.executable]
        bids, output, work = map(str, (p["bids_dir"], p["output_dir"], p["work_dir"]))
        license_path = config.fs_license_file
    else:
        env.update(APPTAINER_CACHEDIR=str(p["cache_dir"] / "apptainer"),
                   APPTAINER_TMPDIR=str(p["work_dir"] / "tmp"))
        prefix = [config.executable, "exec", "--cleanenv", "--containall",
                  "--home", str(p["cache_dir"] / "home") + ":/home/pie",
                  "--bind", str(p["bids_dir"]) + ":/data:ro",
                  "--bind", str(p["output_dir"]) + ":/out",
                  "--bind", str(p["work_dir"]) + ":/work",
                  "--bind", str(p["work_dir"] / "tmp") + ":/tmp",
                  "--bind", str(p["cache_dir"]) + ":/cache",
                  "--env", "TEMPLATEFLOW_HOME=/cache/templateflow,XDG_CACHE_HOME=/cache/xdg,TMPDIR=/tmp",
                  "--env", f"OMP_NUM_THREADS={config.omp_nthreads},OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={config.omp_nthreads}"]
        license_path = "/license.txt" if config.fs_license_file else None
        if license_path:
            prefix += ["--bind", str(Path(config.fs_license_file).resolve()) + ":/license.txt:ro"]
        for name, folder in sorted(config.derivatives.items()):
            prefix += ['--bind', str(Path(folder).expanduser().resolve()) + f':/derivatives/{name}:ro']
        prefix += [str(Path(config.container_image).resolve()), "fmriprep"]
        bids, output, work = "/data", "/out", "/work"
    if version_only:
        return prefix + ["--version"], env
    args = [bids, output, "participant", "--participant-label", *config.participants,
            "-w", work, "--nprocs", str(config.nprocs), "--omp-nthreads", str(config.omp_nthreads),
            "--mem-mb", str(config.memory_mb), "--output-spaces", *config.output_spaces,
            "--random-seed", str(config.random_seed), "--notrack"]
    if not config.surface_reconstruction:
        args += ["--fs-no-reconall"]
    if license_path:
        args += ["--fs-license-file", license_path]
    if config.derivatives:
        args += ['--derivatives', *[
            name + '=' + (str(Path(folder).expanduser().resolve()) if config.backend == 'native'
                          else f'/derivatives/{name}')
            for name, folder in sorted(config.derivatives.items())]]
    return prefix + args + config.extra_args, env


def run_fmriprep(config: FMRIPrepConfig):
    """Execute/resume fMRIPrep with stable configuration and append-only attempts.

    A successful process return is execution completion, not a scientific QC pass.
    Interrupted attempts retain fMRIPrep's work directory for its native resume.
    """
    from .fmri import sha256, write_json
    command, env = build_command(config)
    paths = config.paths()
    for path in [paths["output_dir"], paths["work_dir"], paths["work_dir"] / "tmp",
                 *[paths["cache_dir"] / name for name in ("home", "templateflow", "xdg", "apptainer")]]:
        path.mkdir(parents=True, exist_ok=True)
    version_command, _ = build_command(config, version_only=True)
    version = subprocess.check_output(version_command, env=env, text=True, stderr=subprocess.STDOUT).strip()
    if config.expected_version and not re.search(r"(?<![\d.])" + re.escape(config.expected_version) + r"(?![\d.])", version):
        raise RuntimeError(f"Unexpected fMRIPrep version: {version}")
    serialized = asdict(config)
    # Preserve identities/checkpoints created before optional reuse was added.
    if not config.derivatives:
        serialized.pop('derivatives')
    identity = {"config": serialized, "version": version,
                "container_sha256": sha256(config.container_image) if config.container_image else None}
    if config.derivatives:
        identity['derivative_hashes'] = {
            name: {str(p.relative_to(Path(folder).expanduser().resolve())): sha256(p)
                   for p in sorted(Path(folder).expanduser().resolve().rglob('*')) if p.is_file()}
            for name, folder in sorted(config.derivatives.items())}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    record_dir = paths["output_dir"] / "pie_provenance" / digest
    record_dir.mkdir(parents=True, exist_ok=True)
    settings = paths["work_dir"] / "pie_fmriprep_identity.json"
    if settings.exists() and json.loads(settings.read_text()) != identity:
        raise ValueError("Work directory belongs to different fMRIPrep settings; use a separate work directory")
    if not settings.exists():
        write_json(settings, identity)
    finished = record_dir / "execution_completed.json"
    if finished.exists():
        return json.loads(finished.read_text())
    started = datetime.now(timezone.utc).isoformat()
    attempt = record_dir / ("attempt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".log")
    write_json(record_dir / "configuration.json", dict(identity, argv=command))
    with attempt.open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    record = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "returncode": result.returncode, "log": str(attempt), "identity_sha256": digest,
              "scientific_qc_pass": False, "output_dir": str(paths["output_dir"])}
    write_json(attempt.with_suffix(".json"), record)
    if result.returncode:
        raise RuntimeError(f"fMRIPrep failed ({result.returncode}); inspect {attempt}")
    reports = [paths["output_dir"] / ("sub-" + label + ".html") for label in config.participants]
    if not all(path.is_file() for path in reports):
        raise RuntimeError("fMRIPrep exited successfully but expected participant reports are missing")
    record["reports"] = [str(path) for path in reports]
    write_json(finished, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON fields matching FMRIPrepConfig")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and print argv without running")
    args = parser.parse_args(argv)
    config = FMRIPrepConfig(**json.loads(args.config.read_text()))
    if args.dry_run:
        print(json.dumps({"argv": build_command(config)[0]}, indent=2))
    else:
        print(json.dumps(run_fmriprep(config), indent=2))


if __name__ == "__main__":
    main()
