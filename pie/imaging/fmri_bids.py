"""Explicit, provenance-preserving single-session BIDS export for fMRI.

All acquisition choices are supplied by callers. No outcomes, personal paths,
implicit phase-encoding polarity or largest-output heuristics are used.
"""
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import nibabel as nib
import numpy as np

from .fmri import inspect_nifti, phase_encoding_pair, sha256, write_json


def export_subject(bids_root, subject, t1, t1_sidecar, bold, bold_sidecar,
                   reference=None, reference_sidecar=None, *, discard_seconds=10, reference_volumes=6):
    """Export one selected rest run, optional verified reverse-PE means, and T1.

    fMRI fieldmap means use up to six volumes after ten seconds of initialization
    from each input. Unknown/incompatible reference metadata is retained in the
    provenance record, but cannot become a fieldmap. Original files are untouched.
    """
    if not re.fullmatch(r"[A-Za-z0-9]+", subject):
        raise ValueError('Subject must be an alphanumeric BIDS label')
    if not np.isfinite(discard_seconds) or discard_seconds < 0:
        raise ValueError('discard_seconds must be finite and nonnegative')
    if not isinstance(reference_volumes, int) or reference_volumes < 1:
        raise ValueError('reference_volumes must be a positive integer')
    paths = [Path(p).resolve(strict=True) for p in (t1, t1_sidecar, bold, bold_sidecar)]
    if (reference is None) != (reference_sidecar is None):
        raise ValueError('Reference image and metadata must be supplied together')
    if reference:
        paths += [Path(reference).resolve(strict=True), Path(reference_sidecar).resolve(strict=True)]
    sources = {str(path): sha256(path) for path in paths}
    info = inspect_nifti(bold, bold_sidecar)
    if info['run_class'] != 'rest_candidate' or nib.load(t1).ndim != 3:
        raise ValueError('Need a full 4D resting candidate and unambiguous 3D T1')
    bmeta = json.loads(Path(bold_sidecar).read_text(), strict=False)
    reasons = ['no_reference']
    if reference:
        rmeta = json.loads(Path(reference_sidecar).read_text(), strict=False)
        reference_info = inspect_nifti(reference, reference_sidecar)
        reasons = phase_encoding_pair(info, reference_info)
        rtr = reference_info['tr_seconds']
        if (len(reference_info['shape']) != 4 or rtr is None or
                reference_info['shape'][3] <= math.ceil(discard_seconds / rtr)):
            reasons.append('reference_has_no_post_initialization_volumes')
        if not all(isinstance(m.get('EchoTime'), (int, float)) for m in (bmeta, rmeta)):
            reasons.append('missing_echo_time')
        elif not np.isclose(bmeta['EchoTime'], rmeta['EchoTime'], atol=1e-5, rtol=0):
            reasons.append('different_echo_time')
    root = Path(bids_root).resolve()
    target = root / ('sub-' + subject)
    provenance = root / 'sourcedata' / (subject + '.json')
    identity = {'sources': sources, 'discard_seconds': discard_seconds, 'reference_volumes': reference_volumes}
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        if saved['identity'] != identity:
            raise ValueError('Existing BIDS subject source/settings differ')
        for name, digest in saved['outputs'].items():
            if sha256(root / name) != digest:
                raise ValueError('Existing BIDS output checksum mismatch')
        return saved
    if target.exists():
        raise FileExistsError('Incomplete BIDS subject requires review')
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'dataset_description.json').exists():
        write_json(root / 'dataset_description.json', {'Name': 'PIE resting-state inputs',
                   'BIDSVersion': '1.10.0', 'DatasetType': 'raw', 'Authors': ['PIE research team']})
    (root / 'sourcedata').mkdir(exist_ok=True)
    prefix = 'sub-' + subject
    with tempfile.TemporaryDirectory(prefix='bids-export-', dir=root) as temporary:
        stage = Path(temporary) / prefix
        (stage / 'anat').mkdir(parents=True)
        (stage / 'func').mkdir()
        shutil.copyfile(t1, stage / 'anat' / (prefix + '_T1w.nii.gz'))
        shutil.copyfile(t1_sidecar, stage / 'anat' / (prefix + '_T1w.json'))
        name = prefix + '_task-rest_bold'
        shutil.copyfile(bold, stage / 'func' / (name + '.nii.gz'))
        bmeta = dict(bmeta, TaskName='rest')
        means = []
        if not reasons:
            (stage / 'fmap').mkdir()
            identifier = 'pepolar' + subject
            for direction, source, metadata in [('forward', bold, bmeta), ('reverse', reference, rmeta)]:
                img = nib.load(source)
                discard = math.ceil(discard_seconds / metadata['RepetitionTime'])
                end = min(img.shape[-1], discard + reference_volumes)
                if end <= discard:
                    raise ValueError('Reference has no stable volumes after initialization')
                data = np.asarray(img.dataobj, dtype=np.float32)[..., discard:end].mean(axis=-1)
                if not np.isfinite(data).all():
                    raise ValueError('Nonfinite fieldmap source')
                mean = nib.Nifti1Image(data, img.affine, img.header)
                mean.set_data_dtype(np.float32)
                stem = prefix + '_dir-' + direction + '_epi'
                nib.save(mean, stage / 'fmap' / (stem + '.nii.gz'))
                metadata = {k: v for k, v in metadata.items() if k not in ('SliceTiming', 'TaskName', 'RepetitionTime')}
                metadata.update(B0FieldIdentifier=identifier, IntendedFor='func/' + name + '.nii.gz')
                write_json(stage / 'fmap' / (stem + '.json'), metadata)
                means.append({'source': str(source), 'first_volume_zero_based': discard,
                              'last_volume_exclusive': end, 'output': stem})
            bmeta['B0FieldSource'] = identifier
        write_json(stage / 'func' / (name + '.json'), bmeta)
        stage.rename(target)
    result = {'identity': identity, 'correction_pair_exported': not reasons,
              'unresolved_reasons': sorted(set(reasons)), 'reference_means': means,
              'outputs': {str(p.relative_to(root)): sha256(p) for p in sorted(target.rglob('*')) if p.is_file()}}
    write_json(provenance, result)
    return result
