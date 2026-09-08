"""
fastsurfer.py — run FastSurfer's segmentation-only stream on one T1w NIfTI and parse
its stats. Uses the FastSurfer checkout in PIE/third_party/FastSurfer and the
PIE/venv_imaging interpreter; no FreeSurfer licence is needed for this stream.

The stream is split into (1) FastSurferVINN inference, which needs ~6 GB of GPU memory and
is serialised across worker processes with a file lock, and (2) N4 bias-field correction +
partial-volume-corrected stats, which are CPU-only and run in parallel. This mirrors what
``run_fastsurfer.sh --seg_only --no_cereb --no_hypothal --no_cc`` does.

Outputs per subject (<subjects_dir>/<sid>/): mri/aparc.DKTatlas+aseg.deep.mgz, mri/mask.mgz,
mri/orig_nu.mgz, stats/aseg+DKT.stats (regional volumes + global measures).
"""

import fcntl
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

PIE_ROOT = Path(__file__).resolve().parents[2]
FASTSURFER_HOME = PIE_ROOT / "third_party" / "FastSurfer"
PYTHON = PIE_ROOT / "venv_imaging" / "bin" / "python"
STALL_SECONDS = 420  # no new segmentation for this long -> kill the batch process
STATS_FILE = "stats/aseg+DKT.stats"
SEG_FILE = "mri/aparc.DKTatlas+aseg.deep.mgz"
# label ids run_fastsurfer.sh passes to segstats for the aparc.DKTatlas+aseg segmentation
_SEG_IDS = ("2 4 5 7 8 10 11 12 13 14 15 16 17 18 24 26 28 31 41 43 44 46 47 49 50 51 52 53 54 58 60 63 77 "
            "1002 1003 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 "
            "1023 1024 1025 1026 1027 1028 1029 1030 1031 1034 1035 2002 2003 2005 2006 2007 2008 2009 2010 2011 "
            "2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 "
            "2034 2035").split()


@contextmanager
def _lock(path):
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _run(cmd, log, cwd, timeout=1800, stall=None):
    """Run a FastSurfer script, logging to ``log``. ``timeout`` caps the wall time; ``stall`` is an
    optional callable returning a progress counter, and the process is killed if that counter does
    not change for STALL_SECONDS (FastSurfer's multi-subject mode has been seen to hang after an
    inference)."""
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        env = {**os.environ, "PYTHONPATH": str(cwd)}  # FastSurfer scripts import FastSurferCNN.* from its root
        proc = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd), env=env)
        t0 = last_change = time.time()
        last = stall() if stall else None
        while proc.poll() is None:
            time.sleep(5)
            now = time.time()
            if stall:
                cur = stall()
                if cur != last:
                    last, last_change = cur, now
                elif now - last_change > STALL_SECONDS:
                    proc.kill()
                    proc.wait()
                    raise RuntimeError(f"{Path(cmd[1]).name} stalled for {STALL_SECONDS}s (killed); see {log}")
            if now - t0 > timeout:
                proc.kill()
                proc.wait()
                raise RuntimeError(f"{Path(cmd[1]).name} exceeded {timeout}s (killed); see {log}")
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(cmd[1]).name} exited {proc.returncode}; see {log}")


def _paths(subjects_dir, sid):
    sd = Path(subjects_dir).resolve()
    sub = sd / sid
    for d in ("mri", "stats", "scripts"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    return sd, sub, sub / "mri", sub / STATS_FILE, sub / SEG_FILE, sub / "scripts" / "pie_fastsurfer.log"


def segment(nifti, subjects_dir, sid, threads=4, device="cuda", batch=4, fastsurfer_home=FASTSURFER_HOME, python=PYTHON):
    """FastSurferVINN inference for one scan (GPU, serialised with a file lock). Skips if done."""
    sd, sub, mri, _, seg, log = _paths(subjects_dir, sid)
    if seg.exists():
        return seg
    fs, nifti = Path(fastsurfer_home), Path(nifti).resolve()  # FastSurfer requires absolute paths
    with _lock(sd / ".gpu.lock"):  # one inference on the GPU at a time
        _run([python, fs / "FastSurferCNN" / "run_prediction.py", "--t1", nifti, "--sid", sid, "--sd", sd,
              "--asegdkt_segfile", seg, "--conformed_name", mri / "orig.mgz", "--brainmask_name", mri / "mask.mgz",
              "--aseg_name", mri / "aseg.auto_noCCseg.mgz", "--seg_log", sub / "scripts" / "deep-seg.log",
              "--vox_size", "1", "--batch_size", batch, "--viewagg_device", "auto", "--device", device,
              "--threads", threads], log, fs, timeout=900)
    if not seg.exists():
        raise RuntimeError(f"FastSurferVINN produced no segmentation for {sid}; see {log}")
    return seg


def segment_batch(niftis, subjects_dir, threads=4, device="cuda", batch=4, fastsurfer_home=FASTSURFER_HOME, python=PYTHON):
    """FastSurferVINN inference for many scans in ONE process (models loaded once, ~25 % faster than
    one process per scan). Scan paths must be <dir>/<sid>_T1w.nii.gz; outputs land in <subjects_dir>/<sid>/.
    Returns the sids that now have a segmentation. A failure aborts the rest of the batch (FastSurfer
    stops at the first non-CUDA error), so callers should fall back to ``segment`` for stragglers.
    """
    sd = Path(subjects_dir).resolve()
    sd.mkdir(parents=True, exist_ok=True)
    todo = [Path(n).resolve() for n in niftis if not (sd / Path(n).name.removesuffix("_T1w.nii.gz") / SEG_FILE).exists()]
    if not todo:
        return []
    fs = Path(fastsurfer_home)
    tag = f"batch_{os.getpid()}_{int(time.time())}"
    listing = sd / f".{tag}.csv"
    listing.write_text("\n".join(map(str, todo)) + "\n")
    sids = [n.name.removesuffix("_T1w.nii.gz") for n in todo]
    progress = lambda: sum((sd / sid / SEG_FILE).exists() for sid in sids)
    with _lock(sd / ".gpu.lock"):
        try:
            # relative output names resolve inside <sd>/<sid>/; without --aseg_name FastSurfer writes no aseg
            _run([python, fs / "FastSurferCNN" / "run_prediction.py", "--csv_file", listing, "--sd", sd,
                  "--remove_suffix", "_T1w.nii.gz", "--conformed_name", "mri/orig.mgz",
                  "--brainmask_name", "mri/mask.mgz", "--aseg_name", "mri/aseg.auto_noCCseg.mgz",
                  "--vox_size", "1", "--batch_size", batch,
                  "--viewagg_device", "auto", "--device", device, "--threads", threads], sd / f".{tag}.log", fs,
                 timeout=90 * len(todo) + 300, stall=progress)
        except RuntimeError:
            pass  # stragglers are retried one by one by the caller
    listing.unlink(missing_ok=True)
    return [n.name.removesuffix("_T1w.nii.gz") for n in todo if (sd / n.name.removesuffix("_T1w.nii.gz") / SEG_FILE).exists()]


def complete_segmentation(subjects_dir, sid, fastsurfer_home=FASTSURFER_HOME):
    """Derive mri/mask.mgz and mri/aseg.auto_noCCseg.mgz from the saved aparc.DKTatlas+aseg segmentation,
    exactly as FastSurfer's run_prediction does (its multi-subject mode saves them asynchronously and
    can drop them). Idempotent; returns True if both files exist afterwards."""
    import sys

    import nibabel as nib
    import numpy as np

    mri = Path(subjects_dir).resolve() / sid / "mri"
    seg = mri / "aparc.DKTatlas+aseg.deep.mgz"
    if not seg.exists():
        return False
    if (mri / "mask.mgz").exists() and (mri / "aseg.auto_noCCseg.mgz").exists():
        return True
    if str(fastsurfer_home) not in sys.path:
        sys.path.insert(0, str(fastsurfer_home))
    from FastSurferCNN import reduce_to_aseg as rta

    img = nib.load(seg)
    pred = np.asarray(img.dataobj).astype(np.int16)
    bm = rta.create_mask(pred.copy(), 5, 4)
    if not (mri / "mask.mgz").exists():
        nib.save(nib.MGHImage(bm.astype(np.uint8), img.affine, img.header), mri / "mask.mgz")
    if not (mri / "aseg.auto_noCCseg.mgz").exists():
        aseg = rta.reduce_to_aseg(pred)
        aseg[bm == 0] = 0
        aseg = rta.flip_wm_islands(aseg)
        nib.save(nib.MGHImage(aseg.astype(np.uint8), img.affine, img.header), mri / "aseg.auto_noCCseg.mgz")
    return True


def finish_stats(subjects_dir, sid, threads=4, fastsurfer_home=FASTSURFER_HOME, python=PYTHON):
    """N4 bias correction + partial-volume-corrected stats for a segmented scan (CPU only)."""
    sd, sub, mri, stats, seg, log = _paths(subjects_dir, sid)
    if stats.exists():
        return stats
    if not seg.exists():
        raise RuntimeError(f"no segmentation for {sid}")
    fs = Path(fastsurfer_home)
    if not (mri / "orig_nu.mgz").exists():
        _run([python, fs / "recon_surf" / "N4_bias_correct.py", "--in", mri / "orig.mgz",
              "--rescale", mri / "orig_nu.mgz", "--aseg", mri / "aseg.auto_noCCseg.mgz", "--threads", threads], log, fs, timeout=900)
    _run([python, fs / "FastSurferCNN" / "segstats.py", "--segfile", seg, "--normfile", mri / "orig_nu.mgz",
          "--lut", fs / "FastSurferCNN" / "config" / "FreeSurferColorLUT.txt", "--sd", sd, "--sid", sid,
          "--threads", threads, "--empty", "--excludeid", "0", "--ids", *_SEG_IDS, "--segstatsfile", stats,
          "measures", "--compute", f"Mask({mri / 'mask.mgz'})", "BrainSeg", "BrainSegNotVent", "SupraTentorial",
          "SupraTentorialNotVent", "SubCortGray", "rhCerebralWhiteMatter", "lhCerebralWhiteMatter",
          "CerebralWhiteMatter"], log, fs)
    if not stats.exists():
        raise RuntimeError(f"segstats produced no stats for {sid}; see {log}")
    return stats


def run_fastsurfer(nifti, subjects_dir, sid, threads=4, device="cuda", batch=4,
                   fastsurfer_home=FASTSURFER_HOME, python=PYTHON):
    """Segment one scan and compute its stats (single-scan path; ``run.py`` batches the GPU stage)."""
    if not (Path(subjects_dir).resolve() / sid / STATS_FILE).exists():
        segment(nifti, subjects_dir, sid, threads, device, batch, fastsurfer_home, python)
    return finish_stats(subjects_dir, sid, threads, fastsurfer_home, python)


def parse_stats(stats_path):
    """Parse a FreeSurfer/FastSurfer .stats file into a flat dict.

    Regional rows -> {StructName: Volume_mm3}; '# Measure' lines -> {short name: value}.
    """
    out = {}
    for line in Path(stats_path).read_text().splitlines():
        if line.startswith("# Measure"):
            # "# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1221128.31, mm^3"
            parts = [p.strip() for p in line[len("# Measure"):].split(",")]
            out[parts[1]] = float(parts[3])
        elif line and not line.startswith("#"):
            cols = line.split()
            out[cols[4]] = float(cols[3])
    return out


if __name__ == "__main__":  # self-check on a minimal stats snippet
    import tempfile
    snippet = ("# Measure Mask, MaskVol, Mask Volume, 1601112.000000, mm^3\n"
               "# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange\n"
               "  1   2    257716 261560.204  Left-Cerebral-White-Matter       104.2871     9.4616    21.0000   132.0000   111.0000\n")
    with tempfile.NamedTemporaryFile("w", suffix=".stats", delete=False) as f:
        f.write(snippet)
    d = parse_stats(f.name)
    assert d == {"MaskVol": 1601112.0, "Left-Cerebral-White-Matter": 261560.204}, d
    print("parse_stats self-check OK")
