"""Multi-shell free-water values with raw-tensor and solver-status validation."""
import numpy as np
from .dwi_acquisition import diagnostic_multishell


def fit_multishell_checked(signal, bvals, bvecs):
    """Run unchanged DIPY NLS and reject failed or nonphysical raw solutions.

    Use in one optimizer thread per process: the diagnostic helper temporarily
    intercepts scipy's optimizer to retain pre-clipping tensors and status.
    A DIPY threshold shortcut with no NLS solution is not a validated tissue fit.
    """
    signal, bvals, bvecs = np.asarray(signal, float), np.asarray(bvals, float), np.asarray(bvecs, float)
    if signal.ndim != 2 or signal.shape[1] != len(bvals) or bvecs.shape != (3, len(bvals)):
        raise ValueError('Inconsistent signal/gradient dimensions')
    selected = bvals <= 2050
    shells = sorted(set(np.rint(bvals[selected & (bvals > 50)] / 100).astype(int) * 100))
    if len(shells) < 2 or not np.any(bvals <= 50):
        raise ValueError('Multi-shell fit requires b0 and at least two nonzero shells')
    y = signal[:, selected]
    positive = np.isfinite(y).all(1) & (y > 0).all(1)
    values, diagnostics = diagnostic_multishell(np.maximum(np.nan_to_num(y, nan=1e-6, posinf=1e-6, neginf=1e-6), 1e-6),
                                               bvals[selected], bvecs[:, selected])
    array = lambda key: np.array([v[key] for v in values], float)
    return {'f': array('fw'), 'tissue_md': array('tissue_md'), 'tissue_fa': array('tissue_fa'),
            'physical': positive & np.array([d['accepted'] for d in diagnostics], bool),
            'raw_min_eigenvalue': np.array([d['raw_tensor_min_eigenvalue'] for d in diagnostics]),
            'solver_status': np.array([d['status'] for d in diagnostics], int),
            'diagnostics': diagnostics, 'shells': [int(b) for b in shells]}
