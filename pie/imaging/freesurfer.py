"""freesurfer.py — run FreeSurfer 7.x command-line tools from a local install or an extracted container rootfs.

PIE_FREESURFER_HOME: the FreeSurfer tree (a full 7.4.1 install for SynthSeg, SAMSEG lesions and FastSurfer's surface
stream; the fMRIPrep 25.2.5 rootfs's opt/freesurfer is 7.3.2 without TensorFlow or models). PIE_FS_LICENSE (else
FS_LICENSE): the license file. Default pipelines never import this module; callers opt in.
"""
import os
import signal
import subprocess
from pathlib import Path

HOME = os.environ.get("PIE_FREESURFER_HOME")
LICENSE = os.environ.get("PIE_FS_LICENSE") or os.environ.get("FS_LICENSE")


def fs_env(home=None, license=None):
    """Environment for FreeSurfer subprocesses; raises naming the setting that is missing."""
    home, license = home or HOME, license or LICENSE
    if not home or not (Path(home) / "bin").is_dir():
        raise RuntimeError(f"FreeSurfer not found: set PIE_FREESURFER_HOME (got {home!r})")
    if not license or not Path(license).is_file():
        raise RuntimeError(f"FreeSurfer license not found: set PIE_FS_LICENSE (got {license!r})")
    h, mni = Path(home), Path(home) / "mni"
    path = os.pathsep.join([str(h / "bin"), str(mni / "bin"), os.environ.get("PATH", "")])
    return {**os.environ, "FREESURFER_HOME": str(h), "FREESURFER": str(h), "FS_LICENSE": str(license), "PATH": path,
            "SUBJECTS_DIR": str(h / "subjects"), "MNI_DIR": str(mni), "MINC_BIN_DIR": str(mni / "bin"),
            "MINC_LIB_DIR": str(mni / "lib"), "MNI_DATAPATH": str(mni / "data"),
            "PERL5LIB": str(mni / "lib" / "perl5" / "5.8.5"), "FS_OVERRIDE": "0", "OMP_NUM_THREADS": "1"}


def run(cmd, log, cwd, env=None, timeout=3600):
    """Run one FreeSurfer command, appending its output to ``log``; RuntimeError on a nonzero exit."""
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env if env is not None else fs_env(),
                             stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except BaseException:          # timeout or Ctrl-C: FreeSurfer scripts do their work in child processes
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise
    if rc:
        raise RuntimeError(f"{Path(str(cmd[0])).name} exited {rc}; see {log}")
