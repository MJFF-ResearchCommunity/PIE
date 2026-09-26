"""Versioned anatomical atlases with explicit template-space identity.

The Nilearn Pauli deterministic download is native CIT168, not MNI152.
Never substitute it for this atlas or infer template identity from coordinates.
"""
import hashlib
import json
import shutil
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


def _pinned(url, path, sha256):
    """``path``, downloaded from ``url`` once; raises if its bytes differ from the pinned hash."""
    from .features import _download

    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _download(url, path.with_suffix('.part'))
        path.with_suffix('.part').replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f'{path.name}: checksum differs from the pinned release')
    return path


def templateflow_file(name, cache_dir=CACHE_DIR):
    """Local path of a pinned TemplateFlow file, downloaded once and sha256-checked."""
    return _pinned(TEMPLATEFLOW + name, Path(cache_dir) / name, TEMPLATEFLOW_FILES[name])


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


# Biondetti et al. 2020 (Brain 143:2757, doi:10.1093/brain/awaa216) neuromelanin atlas: their brain-extracted study
# T1 template, the nigral mask split into associative (1), limbic (2) and sensorimotor (3) territories, and the
# background reference (BND) of their SNR = 100 * mean(SN) / mean(BND). The repository declares no licence, so the
# files are fetched at run time and never bundled (verified 26 September 2026).
BIONDETTI = 'https://raw.githubusercontent.com/emmabiondetti/substantia-nigra-neuromelanin/e34cbd55054f22483e85b008f99c3448b978b871/'
BIONDETTI_FILES = {
    'average_nonlin_10.nii.gz': '43ed9ad681af3a106280c81cfe6a0e4e5ccbe69a84321d76e713d4dbdcc1ccf4',
    'SN_ROI_symmetric_three_subdivisions.nii.gz': '54bd3df09f326b868d4dd72701b1d78b83f33e0f2ba2da5731cd0a6d45c3037c',
    'BND_ROI.nii.gz': 'e9b742b222ab6bbd5e43c8558a5c7b0cd699682b1761e3255992fce1b458de6c',
}
BIONDETTI_DIR = Path.home() / '.cache' / 'pie' / 'biondetti'
BIONDETTI_TERRITORIES = {1: 'associative', 2: 'limbic', 3: 'sensorimotor'}
BIONDETTI_BND = 4   # label of the background ROI in the combined MNI image


def biondetti_file(name, cache_dir=BIONDETTI_DIR):
    return _pinned(BIONDETTI + name, Path(cache_dir) / name, BIONDETTI_FILES[name])


def nigral_bridge_qc(label_img, max_mm=4.0):
    """Outcome-blind check of nigral labels (territories 1-3, background 4) mapped to MNI152NLin2009cAsym against the
    CIT168 SNc, the part of the nigra neuromelanin MRI shows (the SNr lies ventrolateral; the study's registration of
    the same atlas put the Biondetti SN 1.0-1.4 mm from the SNc but 3.6-3.7 mm from SNc + SNr): per side, the labelled
    SN's centroid within ``max_mm`` of the SNc's, and no background voxel in the SNc."""
    from nilearn.image import resample_to_img

    cit = resample_to_img(cit168_mni2009c(), label_img, interpolation='nearest', force_resample=True, copy_header=True)
    cit_sn = np.asarray(cit.dataobj) == 7
    lab = np.asarray(label_img.dataobj)
    ijk = np.indices(lab.shape).reshape(3, -1).T
    x = nib.affines.apply_affine(label_img.affine, ijk)[:, 0].reshape(lab.shape)
    out = {}
    for side, half in (('l', x < 0), ('r', x > 0)):
        a = nib.affines.apply_affine(label_img.affine, np.argwhere(np.isin(lab, [1, 2, 3]) & half)).mean(axis=0)
        b = nib.affines.apply_affine(label_img.affine, np.argwhere(cit_sn & half)).mean(axis=0)
        out[f'centroid_mm_{side}'] = float(np.linalg.norm(a - b))
    out['bnd_in_sn'] = int((cit_sn & (lab == BIONDETTI_BND)).sum())
    out['pass'] = bool(out['centroid_mm_l'] <= max_mm and out['centroid_mm_r'] <= max_mm and out['bnd_in_sn'] == 0)
    return out


def _biondetti_transforms(cache_dir=BIONDETTI_DIR, seed=0):
    """Forward transforms, Biondetti template -> MNI152NLin2009cAsym 1 mm brain: their template (negative background
    clipped) registered once with antsRegistrationSyN[s], cached with its ``nigral_bridge_qc`` result; raises while
    that QC fails."""

    import ants

    from .features import _drop_transforms, _seed_ants

    d = Path(cache_dir) / 'to_MNI152NLin2009cAsym'
    fwd, qc_file = [d / '1Warp.nii.gz', d / '0GenericAffine.mat'], d / 'qc.json'
    if not qc_file.exists():
        src = nib.load(biondetti_file('average_nonlin_10.nii.gz', cache_dir))
        moving = ants.from_nibabel_nifti(nib.Nifti1Image(np.clip(np.asarray(src.dataobj, np.float32), 0, None), src.affine))
        _seed_ants(seed)
        reg = ants.registration(ants.image_read(str(mni2009c_brain_1mm())), moving, type_of_transform='antsRegistrationSyN[s]')
        d.mkdir(parents=True, exist_ok=True)
        try:
            for f, dst in zip(reg['fwdtransforms'], fwd):
                shutil.copy(f, dst)
            ants.image_write(reg['warpedmovout'], str(d / 'template_warped.nii.gz'))
        finally:
            _drop_transforms(reg)
        qc = nigral_bridge_qc(_biondetti_labels(fwd, mni2009c_brain_1mm(), cache_dir))
        qc_file.write_text(json.dumps({'qc': qc, 'seed': seed, 'source': BIONDETTI, 'type': 'antsRegistrationSyN[s]',
                                       'target': 'tpl-MNI152NLin2009cAsym_res-01 T1w x brain mask'}, indent=2))
    qc = json.loads(qc_file.read_text())['qc']
    if not qc['pass']:
        raise ValueError(f'Biondetti -> MNI152NLin2009cAsym bridge failed its QC ({qc_file}): {qc}')
    return [str(p) for p in fwd]


def _biondetti_labels(fwd, reference, cache_dir=BIONDETTI_DIR):
    """Territories 1-3 and background 4 pulled once (genericLabel) from Biondetti space onto ``reference``'s grid;
    the nigra wins where the two touch."""
    import ants

    ref = ants.image_read(str(reference))
    sn, bnd = (ants.apply_transforms(ref, ants.image_read(str(biondetti_file(n, cache_dir))), [str(f) for f in fwd],
                                     interpolator='genericLabel').numpy()
               for n in ('SN_ROI_symmetric_three_subdivisions.nii.gz', 'BND_ROI.nii.gz'))
    lab = np.where(sn > 0, np.rint(sn), np.where(bnd > 0, BIONDETTI_BND, 0)).astype(np.int16)
    return nib.Nifti1Image(lab, nib.load(reference).affine)


def biondetti_mni2009c(reference=None, cache_dir=BIONDETTI_DIR, seed=0):
    """Path of the Biondetti territories (1-3) and background (4) in MNI152NLin2009cAsym on ``reference``'s grid (an
    image path; default the 1 mm template), resampled once from the authors' space, not via a second grid."""
    ref = Path(reference) if reference else mni2009c_brain_1mm()
    img = nib.load(ref)
    tag = hashlib.sha256(np.asarray(img.affine, float).tobytes() + repr(img.shape[:3]).encode()).hexdigest()[:10]
    out = Path(cache_dir) / f'biondetti_space-MNI152NLin2009cAsym_grid-{tag}_dseg.nii.gz'
    if not out.exists():
        nib.save(_biondetti_labels(_biondetti_transforms(cache_dir, seed), ref, cache_dir), out)
    return out
