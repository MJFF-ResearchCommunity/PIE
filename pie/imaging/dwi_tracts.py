"""White-matter tract measures on the JHU ICBM-DTI-81 atlas (48 labels).

The atlas most studies use for tract-level FA (Mori 2005, Wakana 2007, Hua 2008; e.g. Droby et al. 2025,
npj Parkinson's Disease). It is fetched at run time from its NeuroVault release (collection 264), not bundled:
that release states no licence.

Template space is never assumed. The atlas is distributed on a generic "MNI" grid whose exact flavour is
undeclared, so labels are brought to each subject by registering the atlas's *own* FA template, which shares
the label grid voxel for voxel, to the subject's FA map (affine then SyN, cross-correlation). No MNI152
variant is involved at any step, which removes the class of error in which an atlas is read in the wrong
template space.

    fetch_jhu(cache_dir)              download + verify the label atlas and its FA template (sha256, grid, laterality)
    map_labels_to_subject(fa, ...)    atlas FA -> subject FA registration; labels pulled with genericLabel interpolation
    registration_qc(...)              correlation of warped template FA with subject FA, label retention, Jacobians
    tract_features(maps, labels)      mean of each map (FA, MD, AD, RD, ...) per tract, with voxel counts

Everything outside the registration is plain NumPy and is exercised by tests on synthetic volumes.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

NEUROVAULT = "https://neurovault.org/media/images/264/"
FILES = {"labels": "JHU-ICBM-labels-1mm.nii.gz", "fa": "JHU-ICBM-FA-1mm.nii.gz"}
# sha256 of the NeuroVault files as first verified on 18 September 2026 (grid 182 x 218 x 182, 1 mm, LAS)
SHA256 = {"labels": "3c2bc4a2aab93d388afc8387a2c0af9c20c92beefb6f28030fb4e8c1ffa8989d",
          "fa": "ae88370bbfd37c72dd3ceeb9c625b2e8d0aba4bb952aa5f6a0e392dd40876b27"}

# ICBM-DTI-81 label names, in label order 1..48 (FSL JHU-labels.xml). For 7..48, odd = right, even = left.
_BILATERAL = [
    "corticospinal_tract", "medial_lemniscus", "inferior_cerebellar_peduncle", "superior_cerebellar_peduncle",
    "cerebral_peduncle", "anterior_limb_internal_capsule", "posterior_limb_internal_capsule",
    "retrolenticular_internal_capsule", "anterior_corona_radiata", "superior_corona_radiata",
    "posterior_corona_radiata", "posterior_thalamic_radiation", "sagittal_stratum", "external_capsule",
    "cingulum_cingulate_gyrus", "cingulum_hippocampus", "fornix_cres_stria_terminalis",
    "superior_longitudinal_fasciculus", "superior_fronto_occipital_fasciculus", "uncinate_fasciculus", "tapetum"]
LABELS = {1: "middle_cerebellar_peduncle", 2: "pontine_crossing_tract", 3: "genu_corpus_callosum",
          4: "body_corpus_callosum", 5: "splenium_corpus_callosum", 6: "fornix_column_body"}
for _i, _name in enumerate(_BILATERAL):
    LABELS[7 + 2 * _i] = _name + "_r"
    LABELS[8 + 2 * _i] = _name + "_l"
assert sorted(LABELS) == list(range(1, 49))


def _download(url, path):
    """Fetch with an explicit User-Agent: NeuroVault's CDN answers 403 to urllib's default one."""
    req = urllib.request.Request(url, headers={"User-Agent": "parkinsons-insight-engine (+https://github.com/MJFF-ResearchCommunity/PIE)"})
    with urllib.request.urlopen(req, timeout=120) as response, open(path, "wb") as out:
        out.write(response.read())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _to_ants(img):
    import ants
    # registration inputs only: real FA and T1 maps carry NaN outside the brain, which ANTs rejects
    data = np.nan_to_num(np.asarray(img.dataobj, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return ants.from_nibabel_nifti(nib.Nifti1Image(data, img.affine))


def _from_ants(ants_img, like):
    """Back to nibabel on the grid of ``like``; refuses any output whose geometry differs from ``like``."""
    import ants
    out = ants.to_nibabel_nifti(ants_img)
    if out.shape[:3] != like.shape[:3] or not np.allclose(out.affine, like.affine, atol=1e-3):
        raise ValueError("resampled image is not on the subject grid; orientation handling is wrong")
    return np.asarray(out.dataobj)


def check_laterality(labels_img, min_separation_mm=4.0):
    """Every label named ``_r`` must lie to the right (+x in scanner RAS) of its ``_l`` partner.

    Returns {pair: right_minus_left_x_mm}; raises if any pair is swapped or not separated.
    """
    data = np.asarray(labels_img.dataobj)
    out = {}
    for k, name in LABELS.items():
        if not name.endswith("_r"):
            continue
        cx = []
        for label in (k, k + 1):
            vox = np.argwhere(data == label)
            if not len(vox):
                raise ValueError(f"label {label} ({LABELS[label]}) is empty")
            cx.append(nib.affines.apply_affine(labels_img.affine, vox)[:, 0].mean())
        out[name[:-2]] = float(cx[0] - cx[1])
    bad = {k: v for k, v in out.items() if v < min_separation_mm}
    if bad:
        raise ValueError(f"left/right labels swapped or not separated: {bad}")
    return out


def fetch_jhu(cache_dir, download=True):
    """The JHU label atlas and its FA template, verified; returns (labels_img, fa_img, provenance)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        path = cache / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            _download(NEUROVAULT + name, path)
        if _sha(path) != SHA256[key]:
            raise ValueError(f"{name}: checksum differs from the verified NeuroVault release")
        paths[key] = path
    labels, fa = nib.load(paths["labels"]), nib.load(paths["fa"])
    if labels.shape != fa.shape or not np.allclose(labels.affine, fa.affine, atol=1e-6):
        raise ValueError("JHU labels and FA template are not on one grid")
    values = np.unique(np.asarray(labels.dataobj))
    if not np.array_equal(values, np.arange(49)):
        raise ValueError("JHU label identity differs from ICBM-DTI-81 (0..48)")
    lateral = check_laterality(labels)
    provenance = {"atlas": "JHU ICBM-DTI-81 white-matter labels", "source": NEUROVAULT, "sha256": dict(SHA256),
                  "registration_target": "atlas FA template on the same grid (no MNI variant assumed)",
                  "laterality_right_minus_left_mm": lateral}
    return labels, fa, provenance


def map_labels_to_subject(subject_fa, atlas_fa, atlas_labels, brain_mask=None, seed=0, syn=True):
    """Register the atlas FA template to the subject FA (affine, then SyN) and pull the labels.

    Returns (labels_img on the subject FA grid, transforms dict). Labels use ``genericLabel`` interpolation.
    """
    import ants

    fixed = _to_ants(subject_fa)
    if brain_mask is not None:
        fixed = fixed * _to_ants(brain_mask)
    moving = _to_ants(atlas_fa)
    kind = "SyN" if syn else "Affine"
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=kind, random_seed=int(seed),
                            syn_metric="CC", syn_sampling=4)
    warped = ants.apply_transforms(fixed=fixed, moving=_to_ants(atlas_labels), transformlist=reg["fwdtransforms"],
                                   interpolator="genericLabel")
    out = nib.Nifti1Image(np.rint(_from_ants(warped, subject_fa)).astype(np.int16), subject_fa.affine)
    warped_fa = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg["fwdtransforms"], interpolator="linear")
    return out, {"fwdtransforms": reg["fwdtransforms"], "warped_template_fa": _from_ants(warped_fa, subject_fa), "type": kind}


def registration_qc(subject_fa, warped_template_fa, subject_labels, atlas_labels, brain_mask,
                    min_correlation=0.5, min_retention=0.5):
    """Checks that must pass before tract values are used.

    template_fa_correlation   Pearson r of warped template FA with subject FA inside the brain mask; on PPMI 2 mm data
                              SyN gave 0.69 where affine alone gave 0.54, with correspondingly higher tract FA
    label_retention           per label, subject-space volume / atlas volume (voxel counts scaled by voxel size)
    """
    s = np.asarray(subject_fa.dataobj, float)
    w = np.asarray(warped_template_fa, float)
    # A real brain mask is required: FA is non-zero noise outside the brain, and averaging over the field of view
    # understates alignment (on PPMI data r fell from 0.69 to 0.50 for the same registration).
    if brain_mask is None:
        raise ValueError("registration_qc needs a brain mask (e.g. dipy median_otsu on the b0)")
    inside = (np.asarray(brain_mask.dataobj) > 0) & np.isfinite(s) & np.isfinite(w)
    r = float(np.corrcoef(s[inside], w[inside])[0, 1])
    sub_vox = float(abs(np.linalg.det(subject_fa.affine[:3, :3])))
    atl_vox = float(abs(np.linalg.det(atlas_labels.affine[:3, :3])))
    sub, atl = np.asarray(subject_labels.dataobj), np.asarray(atlas_labels.dataobj)
    retention = {LABELS[k]: float((sub == k).sum() * sub_vox / max((atl == k).sum() * atl_vox, 1e-9)) for k in LABELS}
    failed = [k for k, v in retention.items() if v < min_retention]
    passed = r >= min_correlation and not failed
    return {"template_fa_correlation": r, "label_retention": retention, "labels_below_retention": failed,
            "qc_pass": bool(passed)}


def tract_features(maps, labels_img, min_voxels=10, fa_floor=None, fa_key="fa"):
    """Mean of every map in ``maps`` ({"fa": img, "md": img, ...}, same grid as the labels) per JHU tract.

    ``fa_floor`` (e.g. 0.2, the TBSS convention) restricts each tract to voxels with FA above it, which reduces
    partial volume with grey matter and CSF; default None averages the whole tract, as most atlas studies do.
    Tracts with fewer than ``min_voxels`` usable voxels are returned as NaN, with their count.
    """
    labels = np.asarray(labels_img.dataobj)
    arrays = {}
    for key, img in maps.items():
        if img.shape[:3] != labels.shape or not np.allclose(img.affine, labels_img.affine, atol=1e-4):
            raise ValueError(f"{key} map and labels are not on one grid")
        arrays[key] = np.asarray(img.dataobj, float)
    if fa_floor is not None and fa_key not in arrays:
        raise ValueError("fa_floor needs the FA map")
    out = {}
    for k, name in LABELS.items():
        sel = labels == k
        for arr in arrays.values():
            sel &= np.isfinite(arr)
        if fa_floor is not None:
            sel &= arrays[fa_key] > fa_floor
        n = int(sel.sum())
        out[f"n_{name}"] = n
        for key, arr in arrays.items():
            out[f"{key}_{name}"] = float(arr[sel].mean()) if n >= min_voxels else float("nan")
    return out


def write_provenance(path, provenance, qc):
    Path(path).write_text(json.dumps({"atlas": provenance, "qc": qc}, indent=1, default=float))
