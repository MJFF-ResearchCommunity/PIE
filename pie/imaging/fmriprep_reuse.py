"""Checked anatomical-only snapshots for fMRIPrep's precomputed input API."""
import json
from pathlib import Path
import re
import shutil
import tempfile

from .fmri import sha256, write_json


def snapshot_anatomical_derivatives(source, target, participants):
    """Copy only explicitly selected subjects' ``anat`` files, never BOLD/fmaps.

    This proves byte identity and anatomical-only scope, not scientific validity.
    The caller must separately verify raw T1 identity, pipeline compatibility,
    successful source execution and anatomical QC before using the snapshot.
    Existing snapshots are verified, never overwritten. Symlinks are rejected.
    """
    source, target = Path(source).resolve(strict=True), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Source and anatomical snapshot must be separate, nonnested paths')
    if not participants or len(set(participants)) != len(participants) or any(
            not re.fullmatch(r'[A-Za-z0-9]+', p) for p in participants):
        raise ValueError('Explicit unique participant labels are required')
    description = source / 'dataset_description.json'
    if description.is_symlink():
        raise ValueError('Symlinked dataset description is not accepted')
    if json.loads(description.read_text()).get('DatasetType') != 'derivative':
        raise ValueError('Source must be a BIDS derivative dataset')
    files = [description]
    for pid in sorted(participants):
        subject = source / ('sub-' + pid)
        anatomies = [subject / 'anat', *sorted(subject.glob('ses-*/anat'))]
        selected = []
        for folder in anatomies:
            if not folder.exists():
                continue
            if folder.is_symlink() or folder.parent.is_symlink() or subject.is_symlink():
                raise ValueError('Symlinked anatomy is not accepted')
            for path in sorted(folder.rglob('*')):
                if path.is_symlink():
                    raise ValueError('Symlinked anatomy is not accepted')
                if path.is_file():
                    # Anatomical folders alone are insufficient: reject misplaced
                    # functional derivatives and transforms from/to BOLD reference.
                    if any(tag in path.name for tag in ('_bold', '_boldref', '_epi', '_confounds',
                                                       'from-bold', 'to-bold')):
                        raise ValueError('Functional derivative found in anatomical folder')
                    selected.append(path)
        if not selected:
            raise ValueError(f'No anatomical derivatives for sub-{pid}')
        files.extend(selected)
    hashes = {str(path.relative_to(source)): sha256(path) for path in files}
    manifest = dict(source=str(source), participants=sorted(participants),
                    anatomical_only=True, scientific_qc_pass=False, files=hashes)
    marker = target / 'sourcedata/pie_anatomical_snapshot.json'
    if target.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != manifest:
            raise ValueError('Existing anatomical snapshot identity changed or incomplete')
        expected = set(hashes) | {'sourcedata/pie_anatomical_snapshot.json'}
        actual = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
        if actual != expected or any(p.is_symlink() for p in target.rglob('*')):
            raise ValueError('Unexpected files in anatomical snapshot')
        if any(sha256(target / key) != value for key, value in hashes.items()):
            raise ValueError('Anatomical snapshot checksum mismatch')
        return manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='anat-snapshot-', dir=target.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        stage.mkdir()
        for key, digest in hashes.items():
            out = stage / key
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / key, out)
            if sha256(out) != digest:
                raise ValueError('Anatomical source changed during snapshot')
        (stage / 'sourcedata').mkdir()
        write_json(stage / 'sourcedata/pie_anatomical_snapshot.json', manifest)
        stage.rename(target)
    return manifest
