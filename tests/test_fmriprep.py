"""Portable invocation contracts; no imaging software or personal data required."""
import json
from pathlib import Path
import subprocess

import pytest

from pie.imaging.fmriprep import FMRIPrepConfig, build_command, run_fmriprep


def config(tmp_path, **kwargs):
    bids = tmp_path / "dataset with spaces"
    (bids / "sub-A01").mkdir(parents=True)
    (bids / "dataset_description.json").write_text('{"Name":"test","BIDSVersion":"1.10.0"}')
    return FMRIPrepConfig(bids_dir=str(bids), output_dir=str(tmp_path / "output"),
                         work_dir=str(tmp_path / "working files"), cache_dir=str(tmp_path / "cache"),
                         participants=["A01"], **kwargs)


def test_native_paths_are_single_arguments_and_dry_run_has_no_writes(tmp_path):
    c = config(tmp_path)
    argv, env = build_command(c)
    assert argv[:4] == ["fmriprep", c.bids_dir, c.output_dir, "participant"]
    assert argv[argv.index("-w") + 1] == c.work_dir
    assert env["TMPDIR"] == str(Path(c.work_dir) / "tmp")
    assert env["TEMPLATEFLOW_HOME"] == str(Path(c.cache_dir) / "templateflow")
    assert "--fs-no-reconall" in argv and "--notrack" in argv
    assert not Path(c.work_dir).exists() and not Path(c.output_dir).exists()


@pytest.mark.parametrize('backend', ['native', 'apptainer'])
def test_caller_paths_need_no_mount_or_device_discovery(tmp_path, monkeypatch, backend):
    image = tmp_path / 'image.sif'
    image.write_bytes(b'synthetic container')
    c = config(tmp_path, backend=backend, container_image=str(image))
    def forbidden(*args, **kwargs):
        raise AssertionError('Configuration must not invoke mount/device discovery')
    monkeypatch.setattr(subprocess, 'check_output', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    argv, env = build_command(c)
    assert argv and env['TMPDIR'] == str(Path(c.work_dir) / 'tmp')
    assert not Path(c.work_dir).exists()


def test_container_maps_user_paths_and_caches(tmp_path):
    image = tmp_path / "container image.sif"
    image.write_bytes(b"test")
    c = config(tmp_path, backend="apptainer", executable="/opt/tools/apptainer", container_image=str(image))
    argv, env = build_command(c)
    assert argv[:2] == [c.executable, "exec"]
    assert c.bids_dir + ":/data:ro" in argv
    assert c.work_dir + ":/work" in argv
    assert argv[argv.index(str(image)) + 1:][:4] == ["fmriprep", "/data", "/out", "participant"]
    assert env["APPTAINER_TMPDIR"].startswith(c.work_dir)
    assert env["APPTAINER_CACHEDIR"].startswith(c.cache_dir)
    assert build_command(c, version_only=True)[0][-1] == "--version"


@pytest.mark.parametrize("change", [
    {"participants": ["../person"]}, {"participants": ["A01", "A01"]},
    {"participants": ["missing"]}, {"nprocs": 0}, {"omp_nthreads": 8},
    {"backend": "other"}, {"backend": "apptainer"},
    {"surface_reconstruction": True}, {"extra_args": ["--work-dir=/tmp"]},
])
def test_invalid_settings_rejected(tmp_path, change):
    c = config(tmp_path)
    for key, value in change.items():
        setattr(c, key, value)
    with pytest.raises(ValueError):
        build_command(c)


def test_paths_cannot_overwrite_input(tmp_path):
    c = config(tmp_path)
    c.output_dir = str(Path(c.bids_dir) / "outputs")
    with pytest.raises(ValueError, match="separate"):
        build_command(c)


def test_process_success_is_not_scientific_qc_and_repeat_reuses(tmp_path, monkeypatch):
    c = config(tmp_path, expected_version="25.2.5")
    calls = []
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "fMRIPrep v25.2.5\n")
    def execute(args, **kwargs):
        calls.append(args)
        (Path(c.output_dir) / "sub-A01.html").write_text("test report")
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", execute)
    result = run_fmriprep(c)
    identity = json.loads((Path(c.work_dir) / 'pie_fmriprep_identity.json').read_text())
    assert 'derivatives' not in identity['config']  # legacy checkpoint compatibility
    assert result["returncode"] == 0 and result["scientific_qc_pass"] is False
    assert run_fmriprep(c) == result and len(calls) == 1
    c.nprocs = 2
    with pytest.raises(ValueError, match="different"):
        run_fmriprep(c)


def test_wrong_version_or_failed_process_never_marks_complete(tmp_path, monkeypatch):
    c = config(tmp_path, expected_version="25.2.5")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "fMRIPrep v25.2.50")
    with pytest.raises(RuntimeError, match="version"):
        run_fmriprep(c)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "fMRIPrep v25.2.5")
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 2))
    with pytest.raises(RuntimeError, match="failed"):
        run_fmriprep(c)
    assert not list(Path(c.output_dir).rglob("execution_completed.json"))


@pytest.mark.parametrize('backend', ['native', 'apptainer'])
def test_reuse_paths_are_explicit_read_only_and_checked(tmp_path, backend):
    image = tmp_path / 'image.sif'
    image.write_bytes(b'image')
    reuse = tmp_path / 'anatomical derivative snapshot'
    reuse.mkdir()
    (reuse / 'dataset_description.json').write_text('{"DatasetType":"derivative"}')
    c = config(tmp_path, backend=backend, container_image=str(image), derivatives={'anatomy': str(reuse)})
    args, _ = build_command(c)
    if backend == 'apptainer':
        assert str(reuse) + ':/derivatives/anatomy:ro' in args
        assert 'anatomy=/derivatives/anatomy' in args
    else:
        assert 'anatomy=' + str(reuse) in args
    c.output_dir = str(reuse / 'new_outputs')
    with pytest.raises(ValueError, match='separate'):
        build_command(c)


def test_changed_derivative_invalidates_work_identity(tmp_path, monkeypatch):
    reuse = tmp_path / 'anatomy'
    reuse.mkdir()
    (reuse / 'dataset_description.json').write_text('{"DatasetType":"derivative"}')
    c = config(tmp_path, derivatives={'anatomy': str(reuse)})
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: 'fMRIPrep v25.2.5')
    def execute(args, **kwargs):
        (Path(c.output_dir) / 'sub-A01.html').write_text('report')
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, 'run', execute)
    run_fmriprep(c)
    (reuse / 'new_T1w.nii.gz').write_bytes(b'changed source')
    with pytest.raises(ValueError, match='different'):
        run_fmriprep(c)
