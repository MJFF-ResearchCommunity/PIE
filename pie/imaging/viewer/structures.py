"""Display-only boundaries of measured, participant-specific segmentation labels.

No template, registration, cortical thickness estimation, or SPECT processing.
Vertices are exported in the atlas's scanner RAS frame, never FreeSurfer tkRAS.
"""
from __future__ import annotations

import hashlib
import json

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from .images import label_table, same_grid, validate_image


def boundary_mesh(mask, affine):
    """Half-voxel isosurface, padded to retain closed boundaries at image edges."""
    points = np.argwhere(mask)
    if len(points) < 8:
        raise ValueError("Too few labelled voxels for a boundary")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    crop = np.pad(mask[tuple(slice(a, b) for a, b in zip(lo, hi))], 1)
    vertices, faces, _, _ = marching_cubes(crop.astype(np.float32), level=0.5)
    vertices = nib.affines.apply_affine(affine, vertices + lo - 1).astype(np.float32)
    # A handedness-changing affine reverses winding as well as coordinates.
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]
    return vertices, faces.astype(np.int32)


def prepare_structures(store, scan):
    if not scan.atlas or scan.atlas_name != "DKT + aseg" or not scan.atlas_lut:
        raise ValueError("Structure exploration requires the participant's named DKT + aseg segmentation")
    atlas = nib.load(scan.atlas)
    validate_image(atlas, label=True)
    if not same_grid(atlas, nib.load(scan.path)):
        raise ValueError("Segmentation must match the anatomical reference grid")
    signature = "structures-v1|" + json.dumps(scan.public(), sort_keys=True) + "|" + "|".join(
        f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in (scan.atlas, scan.atlas_lut)
    )
    key = hashlib.sha256(signature.encode()).hexdigest()[:24]
    with store._lock:
        folder = store.cache / f"structures-{key}"
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder / "structures.json"
        if not manifest.exists():
            data = np.asarray(atlas.dataobj)
            if np.any(~np.isfinite(data)) or np.any(data != np.rint(data)):
                raise ValueError("Segmentation contains non-integer labels")
            table = label_table(scan.atlas_lut)
            ids = {int(i) for i in np.unique(data)}
            groups = [
                ("left", "Left hemisphere shell", [2, 3] + [i for i in ids if 1000 <= i < 2000], [189, 199, 190]),
                ("right", "Right hemisphere shell", [41, 42] + [i for i in ids if 2000 <= i < 3000], [189, 199, 190]),
                ("context", "Cerebellum and brainstem", [7, 8, 16, 46, 47], [160, 175, 165]),
            ]
            for i in (10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58):
                if i in ids and i in table:
                    groups.append((str(i), table[i]["name"].replace("-", " "), [i], table[i]["color"]))
            meshes = []
            for group, name, labels, color in groups:
                mask = np.isin(data, labels)
                if np.count_nonzero(mask) < 8:
                    continue
                vertices, faces = boundary_mesh(mask, atlas.affine)
                filename = f"structure-{group}.surf.gii"
                gii = nib.gifti.GiftiImage(darrays=[
                    nib.gifti.GiftiDataArray(vertices, intent="NIFTI_INTENT_POINTSET",
                        coordsys=nib.gifti.GiftiCoordSystem(dataspace=1, xformspace=1, xform=np.eye(4))),
                    nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"),
                ])
                nib.save(gii, folder / filename)
                meshes.append({"key": group, "name": name, "filename": filename, "color": color,
                               "region_ids": [i for i in labels if i in ids]})
            manifest.write_text(json.dumps(meshes))
        meshes = json.loads(manifest.read_text())
        return {"reference_id": scan.id, "space": scan.space, "fingerprint": key,
                "note": "Native-voxel segmentation boundaries, not validated pial surfaces or cortical thickness. Automated labels need visual review.",
                "meshes": [{**m, "url": store.asset(folder / m["filename"])} for m in meshes]}
