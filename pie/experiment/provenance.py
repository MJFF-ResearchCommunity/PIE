"""Run identity: content hashes, JSON that survives numpy, and manifests.

Every reproducible experiment ends up writing the same record — when it ran, which
inputs it read, which code produced it, what it wrote — and every study re-implements
it slightly differently. These are the primitives for that record.

    from pie.experiment import provenance as prov

    started = time.time()
    ...
    prov.write_manifest(out_dir, inputs=[cohort_csv], code=Path(__file__).parent,
                        started=started, seed=20260913, datasets=summary)

`sha256` is chunked, so hashing a multi-gigabyte NIfTI costs no memory. `save_json`
writes NaN and infinities as `null` rather than the non-standard `NaN` token that trips
strict readers.
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

_CHUNK = 1024 * 1024


def utc_now():
    """Timezone-explicit timestamp; a naive local timestamp is not provenance."""
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


file_sha256 = sha256   # name used by the study scripts this was extracted from


def jsonable(value):
    """Plain-Python copy of a structure holding numpy scalars; non-finite floats become None."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value, indent=2):
    """Write `value` as strict JSON. NaN and ±inf become null (via `jsonable`).

    allow_nan=False is the backstop for a non-finite value `jsonable` does not reach;
    anything else it does not recognise is written as `str(value)`.
    """
    Path(path).write_text(json.dumps(jsonable(value), indent=indent, default=str, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def code_hashes(source, pattern="*.py"):
    """Hash every source file of a run: a directory's sources, or an explicit list of files."""
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        paths = sorted(Path(source).glob(pattern))
        return {p.name: sha256(p) for p in paths}
    paths = [Path(source)] if isinstance(source, (str, Path)) else [Path(p) for p in source]
    return {str(p): sha256(p) for p in paths if p.is_file()}


def environment(packages=("numpy", "scipy", "pandas", "sklearn", "joblib", "nibabel", "torch"), binaries=()):
    """Interpreter, kernel and the versions of the packages a result actually depends on.

    Missing packages are simply absent; recording `null` for an uninstalled package
    would suggest it was consulted.
    """
    versions = {}
    for name in packages:
        try:
            module = __import__(name)
        except Exception:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    record = {"python": sys.version, "platform": platform.platform(), "packages": versions}
    if binaries:
        record["binaries"] = {str(b): sha256(b) for b in binaries if Path(b).is_file()}
    return record


def _relative(path, out_dir):
    """`path` relative to `out_dir`, POSIX-style (`../x` for files outside it)."""
    return Path(os.path.relpath(Path(path).absolute(), out_dir.absolute())).as_posix()


def write_manifest(out_dir, inputs=(), code=None, outputs=None, started=None,
                   name="manifest.json", include_environment=True, **extra):
    """Write `out_dir/name` describing this run, and return the record.

    inputs   paths read; each must be an existing file. A missing one raises rather than
             silently dropping out of the record
    code     directory or list of source files whose content defines the analysis
    outputs  paths written (each must exist); None means every file under out_dir,
             recursively, except the manifest. Keyed by path relative to out_dir, so
             subdirectories and files outside out_dir re-verify correctly
    started  time.time() at the start of the run, to record wall-clock seconds
    extra    anything else the study wants on the record (seed, counts, selections)
    """
    out_dir = Path(out_dir)
    inputs = list(inputs)
    outputs = None if outputs is None else list(outputs)
    missing = [str(p) for p in [*inputs, *(outputs or [])] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest paths that are not existing files: {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"created_utc": utc_now(), "inputs_sha256": {str(p): sha256(p) for p in inputs}}
    if code is not None:
        record["code_sha256"] = code_hashes(code)
    if include_environment:
        record["environment"] = environment()
    if started is not None:
        record["seconds"] = round(time.time() - started, 1)
    record.update(extra)
    # Outputs last: hashing the manifest into itself is impossible, so it is always excluded.
    if outputs is None:
        outputs = [p for p in sorted(out_dir.rglob("*")) if p.is_file() and p != out_dir / name]
    record["outputs_sha256"] = {_relative(p, out_dir): sha256(p) for p in outputs}
    save_json(out_dir / name, record)
    return record


def verify_manifest(out_dir, name="manifest.json"):
    """Re-hash what a manifest claims and return the paths that no longer match.

    Empty result means the run's inputs and outputs are byte-identical to when it ran.
    Output keys are paths relative to out_dir (older manifests keyed top-level files by
    bare name, which is the same thing). Code hashes are not re-checked here.
    """
    out_dir = Path(out_dir)
    record = read_json(out_dir / name)
    changed = []
    for path, digest in (record.get("inputs_sha256") or {}).items():
        if not Path(path).is_file() or sha256(path) != digest:
            changed.append(path)
    for filename, digest in (record.get("outputs_sha256") or {}).items():
        target = Path(os.path.normpath(out_dir / filename))
        if not target.is_file() or sha256(target) != digest:
            changed.append(str(target))
    return changed
