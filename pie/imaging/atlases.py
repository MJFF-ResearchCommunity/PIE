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
MNI_TEMPLATE_STEM = 'MNI152NLin2009cAsym_brain_2mm'


def mni2009c_template_metadata():
    return json.loads((ATLAS_DIR / (MNI_TEMPLATE_STEM + '.json')).read_text())


def mni2009c_template():
    """The verified reference matching CIT168, independent of Nilearn defaults."""
    meta = mni2009c_template_metadata()
    if meta['space'] != MNI_SPACE:
        raise ValueError('CIT168 atlas and registration template spaces differ')
    path = ATLAS_DIR / meta['filename']
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Registration template checksum mismatch')
    image = nib.load(path)
    if list(image.shape) != meta['shape'] or not np.allclose(image.affine, meta['affine'], rtol=0, atol=1e-6):
        raise ValueError('Registration template grid mismatch')
    return image


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
            'atlas_sha256': meta['sha256'], 'atlas_probability_threshold': meta['probability_threshold'],
            'registration_reference_space': MNI_SPACE,
            'registration_reference_sha256': mni2009c_template_metadata()['sha256']}


# TemplateFlow files in the atlas's own space, fetched on first use and sha256-checked (verified 25 September 2026)
TEMPLATEFLOW = 'https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/'
TEMPLATEFLOW_FILES = {
    'tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz': '1f27aabea9f7183dc0c69dafa71e1787b921ca778d655d9c1bf302b273d5627a',
    'tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz': 'e40bb1816736504d4c25abc243c1c8503df1d308f64c048318aa4b829e09a7ae',
    'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz':
        'e5dfdc5674fe6122609fa8d223d7d7c17f989ff1b03d7746559c95b11da1d264',
    'tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.tsv':
        '91fe503df22267bb6c9ab3b9e9e82eec0e08f496d4f9dcb9661e67d1abb14f07',
}
CACHE_DIR = Path.home() / '.cache' / 'pie' / 'templateflow'


def templateflow_file(name, cache_dir=CACHE_DIR):
    """Local path of a pinned TemplateFlow file, downloaded once; raises if its bytes differ from the pinned hash."""
    from .features import _download

    path = Path(cache_dir) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _download(TEMPLATEFLOW + name, path.with_suffix('.part'))
        path.with_suffix('.part').replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != TEMPLATEFLOW_FILES[name]:
        raise ValueError(f'{name}: checksum differs from the pinned TemplateFlow release')
    return path


def mni2009c_brain_1mm(cache_dir=CACHE_DIR):
    """Path of the brain-masked 1 mm MNI152NLin2009cAsym T1w: the deformable (SyN) target of ``dwi_refine`` and
    ``nm_template``, in the space of the bundled CIT168 atlas (nilearn's 1 mm template is the 2009a release)."""
    out = Path(cache_dir) / 'MNI152NLin2009cAsym_res-01_brain.nii.gz'
    t1 = nib.load(templateflow_file('tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz', cache_dir))
    mask = nib.load(templateflow_file('tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz', cache_dir))
    if not out.exists():
        brain = np.asarray(t1.dataobj, np.float32) * (np.asarray(mask.dataobj) > 0)
        nib.save(nib.Nifti1Image(brain, t1.affine), out)
    return out


def schaefer400_mni2009c(cache_dir=CACHE_DIR):
    """Schaefer et al. 2018 400-parcel, 7-network cortical parcellation in MNI152NLin2009cAsym (2 mm), the default
    cortical atlas of XCP-D's "4S" set. Returns (label image, {parcel id: network}) in the form
    ``fmri_connectivity.extract_connectivity`` takes, networks being Vis SomMot DorsAttn SalVentAttn Limbic Cont Default."""
    import pandas as pd

    img = nib.load(templateflow_file('tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.nii.gz', cache_dir))
    table = pd.read_csv(templateflow_file('tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.tsv', cache_dir), sep='\t')
    networks = {int(i): name.split('_')[2] for i, name in zip(table['index'], table['name'])}   # 7Networks_LH_Vis_1 -> Vis
    if set(np.unique(np.asarray(img.dataobj))) - {0} != set(networks):
        raise ValueError('Schaefer label image and table differ')
    return img, networks
