"""Gradient rebuild for Siemens series with stripped diffusion tags (synthetic headers and geometry)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pie.imaging import dwi_gradients as g


class H(dict):
    def get(self, k, default=None):
        return super().get(k, default)


def _session(dirs_world, rot_deg, flip_det=False):
    """A session whose slices are rotated by ``rot_deg`` about z: its FSL bvecs for directions fixed in patient space."""
    r = Rotation.from_euler("z", rot_deg, degrees=True).as_matrix()
    if flip_det:
        r = r @ np.diag([-1.0, 1, 1])
    aff = np.eye(4)
    aff[:3, :3] = r * 2.0
    n = dirs_world.shape[1]
    vt = g.volume_table([H(AcquisitionNumber=1, SequenceName="*ep_b0")] +
                        [H(AcquisitionNumber=i + 2, SequenceName=f"*ep_b1000#{i + 1}") for i in range(n)])
    world = np.c_[np.zeros(3), dirs_world]
    return vt, g._from_frame(world, aff, "patient"), aff


def test_patient_frame_directions_are_rebuilt_exactly_and_the_image_frame_is_rejected():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(3, 12))
    d /= np.linalg.norm(d, axis=0)
    intact = [_session(d, a, flip) for a, flip in ((0, False), (15, False), (-20, True), (30, False), (5, True))]
    assert max(g.leave_one_out(intact, "patient")) < 1e-6
    assert max(g.leave_one_out(intact, "image")) > 10                 # slice orientations differ: not this frame
    ref = g.reference_directions(intact, "patient")
    vt, bvec, aff = _session(d, 42.0)
    bvals, rebuilt = g.rebuild(vt, aff, ref)
    assert bvals.tolist() == [0] + [1000] * 12 and g.angle_deg(rebuilt[:, 1:], bvec[:, 1:]).max() < 1e-6 and ref["spread_deg"] < 1e-6


def test_names_without_an_index_or_disagreeing_files_are_refused():
    with pytest.raises(ValueError, match="no b-value"):
        g.volume_table([H(AcquisitionNumber=1, SequenceName="*epse2d1_128")])
    with pytest.raises(ValueError, match="disagree"):
        g.volume_table([H(AcquisitionNumber=1, SequenceName="*ep_b1000#1"), H(AcquisitionNumber=1, SequenceName="*ep_b1000#2")])
