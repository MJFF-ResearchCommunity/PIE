import hashlib
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from pie.imaging.viewer.catalog import Scan
from pie.imaging.viewer.images import ImageStore
from pie.imaging.viewer.images import write_volume
from pie.imaging.viewer.structures import boundary_mesh, prepare_structures


def test_boundary_preserves_oblique_ras_and_edge_voxels():
    mask = np.zeros((8, 9, 10), bool)
    mask[0:3, 2:5, 4:7] = True
    affine = np.array([[-2., 0, 0, 20], [0, 0, -3, 15], [0, 4, 0, -5], [0, 0, 0, 1]])
    vertices, faces = boundary_mesh(mask, affine)
    vox = nib.affines.apply_affine(np.linalg.inv(affine), vertices)
    np.testing.assert_allclose(vox.min(axis=0), [-.5, 1.5, 3.5])
    np.testing.assert_allclose(vox.max(axis=0), [2.5, 4.5, 6.5])
    assert faces.min() == 0 and faces.max() < len(vertices)


def test_prepared_atlas_has_integer_label_intent(tmp_path):
    path = tmp_path / "input.nii.gz"
    data = np.ones((3, 4, 5), np.uint16) * 11
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    out = write_volume(path, tmp_path / "atlas.nii.gz", label=True)
    assert out.header.get_intent()[0] == "label"
    np.testing.assert_array_equal(out.get_fdata(), data)


def test_structures_are_label_specific_cached_and_source_is_unchanged(tmp_path):
    data = np.zeros((12, 12, 12), np.uint16)
    data[1:5, 2:6, 3:7] = 11
    data[7:10, 2:6, 3:7] = 50
    data[2:5, 8:11, 3:7] = 1002
    path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    lut = Path(__file__).parents[1] / "pie/imaging/viewer/atlas.tsv"
    scan = Scan("mri-test", "001", "MRI", "2020-01-01", "BL", "test", path, "native", atlas=path, atlas_lut=lut, atlas_name="DKT + aseg")
    store = ImageStore(tmp_path / "cache")
    a = prepare_structures(store, scan)
    assert {m["key"] for m in a["meshes"]} == {"left", "11", "50"}
    for m in a["meshes"]:
        assert m["region_ids"] == ([1002] if m["key"] == "left" else [int(m["key"])])
        mesh = nib.load(store.assets[m["url"].split("/")[-2]])
        assert mesh.darrays[0].coordsys.xformspace == 1
    assert a == prepare_structures(store, scan)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    scan.atlas_name = "Unknown"
    with pytest.raises(ValueError, match="named"):
        prepare_structures(store, scan)
