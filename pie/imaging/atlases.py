"""Versioned anatomical atlases with explicit template-space identity.

The Nilearn Pauli deterministic download is native CIT168, not MNI152.
Never substitute it for this atlas or infer template identity from coordinates.
"""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ATLAS_DIR = Path(__file__).with_name('data') / 'atlases'
CIT168_STEM = 'CIT168_v1_MNI152NLin2009cAsym_det25'
MNI_SPACE = 'MNI152NLin2009cAsym'


def cit168_metadata():
    return json.loads((ATLAS_DIR / (CIT168_STEM + '.json')).read_text())


def cit168_mni2009c(expected_space=MNI_SPACE):
    """Load the author's MNI2009c atlas, checking file, grid and label identity."""
    meta = cit168_metadata()
    if expected_space != MNI_SPACE or meta['space'] != expected_space:
        raise ValueError('CIT168 atlas and registration template spaces differ')
    path = ATLAS_DIR / meta['filename']
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('CIT168 atlas checksum mismatch')
    image = nib.load(path)
    if list(image.shape) != meta['shape'] or not np.allclose(image.affine, meta['affine'], rtol=0, atol=1e-6):
        raise ValueError('CIT168 atlas grid mismatch')
    labels = np.unique(np.asarray(image.dataobj))
    if not np.array_equal(labels, np.arange(len(meta['labels']) + 1)):
        raise ValueError('CIT168 atlas label identity mismatch')
    return image


def cit168_provenance():
    meta = cit168_metadata()
    return {'atlas_space': meta['space'], 'atlas_version': meta['version'],
            'atlas_sha256': meta['sha256'], 'atlas_probability_threshold': meta['probability_threshold']}
