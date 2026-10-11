"""
dwi_gradients.py — rebuild diffusion gradients for Siemens series whose export stripped the private diffusion tags.

Some PPMI 2.0 Siemens DTI downloads lack the CSA headers (0029,1010 / 0029,1020) and the b-value / direction tags
(0019,100C / 0019,100E), so dcm2niix writes no bval/bvec and the session is lost. ``SequenceName`` often survives:
``*ep_b1000#11`` is b = 1000, direction 11 of the protocol's table; ``*ep_b0`` a b0. The direction vectors come from
intact sessions of the same protocol, in the frame in which they are shown to be the same across sessions (patient or
slice/image), as pre-registered (2026-09-28 DWI gate, Addendum 1):
- on >= 5 intact sessions, each rebuilt leave-one-out must match its real vectors within 1 deg;
- MD from the rebuilt gradients must correlate with the real MD at r >= 0.99;
otherwise recovered sessions are not used. Series whose SequenceName carries no index (XA30 ``*epse2d1_128``) cannot be
rebuilt this way.

    vt = volume_table(series_headers(zip_path, prefix))          # per volume: b-value, direction index
    ref = reference_directions([(vt_i, bvec_i, affine_i), ...], frame="patient")
    bvals, bvecs = rebuild(vt, affine, ref)
"""
import io
import re
import zipfile

import numpy as np
import pandas as pd

SEQ = re.compile(r"ep_b(\d+)(?:t)?(?:#(\d+))?")


def series_headers(zip_path, prefix):
    """Every DICOM header of one series (no pixel data), in member order."""
    import pydicom

    with zipfile.ZipFile(zip_path) as z:
        return [pydicom.dcmread(io.BytesIO(z.read(n)), force=True, stop_before_pixels=True)
                for n in sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".dcm"))]


def volume_table(headers):
    """One row per volume (``AcquisitionNumber``, the order dcm2niix stacks classic Siemens DICOM): ``b`` and direction
    ``k`` (0 for b0) from ``SequenceName``. Raises when a volume's files disagree or a name carries no index."""
    rows = {}
    for ds in headers:
        acq = int(ds.get("AcquisitionNumber", 0) or 0)
        m = SEQ.search(str(ds.get("SequenceName", "")))
        if m is None:
            raise ValueError(f"SequenceName {ds.get('SequenceName')!r} carries no b-value: gradients cannot be rebuilt")
        b, k = int(m.group(1)), int(m.group(2) or 0)
        if acq in rows and rows[acq] != (b, k):
            raise ValueError(f"volume {acq}: files disagree on SequenceName")
        rows[acq] = (b, k)
    t = pd.DataFrame([(a, b, k) for a, (b, k) in sorted(rows.items())], columns=["acq", "b", "k"])
    if ((t["b"] > 50) & (t["k"] == 0)).any():
        raise ValueError("a diffusion-weighted volume without a direction index")
    return t


def _to_frame(bvec, affine, frame):
    """FSL bvecs (image axes, x flipped when the affine's determinant is positive) as unit vectors in ``frame``:
    "image" keeps the image axes; "patient" rotates them by the affine's direction cosines into world (RAS)."""
    v = np.asarray(bvec, float).reshape(3, -1).copy()
    if frame == "image":
        return v
    r = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    if np.linalg.det(r) > 0:
        v[0] *= -1
    return r @ v


def _from_frame(v, affine, frame):
    if frame == "image":
        return v
    r = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    out = r.T @ v
    if np.linalg.det(r) > 0:
        out[0] *= -1
    return out


def _axis_mean(vs):
    """Mean of axes (g and -g are one direction): sign-aligned to the first, normalised."""
    vs = np.asarray(vs, float)
    vs = vs * np.sign(vs @ vs[0])[:, None]
    m = vs.mean(axis=0)
    return m / np.linalg.norm(m)


def angle_deg(a, b):
    """Angle between axes (unsigned), degrees, column-wise."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.abs((a * b).sum(axis=0)) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def reference_directions(intact, frame="patient"):
    """{(b, k): unit vector in ``frame``} from intact sessions [(volume_table, bvec 3 x n, affine)], with the largest
    angle of any session's vector from the mean (``spread_deg``) as the invariance check of that frame."""
    by = {}
    for vt, bvec, affine in intact:
        v = _to_frame(bvec, affine, frame)
        if v.shape[1] != len(vt):
            raise ValueError("bvec and volume table differ in length")
        for (b, k), col in zip(zip(vt["b"], vt["k"]), v.T):
            if b > 50:
                by.setdefault((b, k), []).append(col / np.linalg.norm(col))
    ref = {key: _axis_mean(vs) for key, vs in by.items()}
    spread = max((float(angle_deg(np.array(vs).T, np.repeat(ref[key][:, None], len(vs), 1)).max()) for key, vs in by.items()), default=np.nan)
    return {"frame": frame, "vectors": ref, "spread_deg": spread, "n_sessions": len(intact)}


def rebuild(vt, affine, ref):
    """(bvals, bvecs in FSL image axes) for a stripped series from its volume table and ``reference_directions``."""
    bvals = vt["b"].to_numpy(float)
    v = np.zeros((3, len(vt)))
    for i, (b, k) in enumerate(zip(vt["b"], vt["k"])):
        if b > 50:
            if (b, k) not in ref["vectors"]:
                raise ValueError(f"direction ({b}, {k}) not in the reference table")
            v[:, i] = ref["vectors"][(b, k)]
    out = _from_frame(v, affine, ref["frame"])
    out[:, bvals <= 50] = 0
    return bvals, out


def leave_one_out(intact, frame):
    """Each intact session rebuilt from the others: the largest angle (deg) to its real vectors, per session."""
    worst = []
    for i, (vt, bvec, affine) in enumerate(intact):
        ref = reference_directions([s for j, s in enumerate(intact) if j != i], frame)
        _, v = rebuild(vt, affine, ref)
        dw = vt["b"].to_numpy() > 50
        worst.append(float(angle_deg(v[:, dw], np.asarray(bvec, float)[:, dw]).max()))
    return worst
