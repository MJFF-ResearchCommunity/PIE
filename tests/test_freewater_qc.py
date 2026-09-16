import numpy as np
from pie.imaging import freewater_qc


def test_rejects_clipped_tensor_and_nonconvergence_even_with_plausible_fraction(monkeypatch):
    # All displayed FW/FA values appear plausible; raw physicality/status differ.
    values = [{'fw':.2,'tissue_fa':.4,'tissue_md':.0007} for _ in range(3)]
    diagnostics = [dict(accepted=ok,raw_tensor_min_eigenvalue=ev,status=st)
                   for ok,ev,st in [(True,.0002,1),(False,-.0001,1),(False,.0002,5)]]
    monkeypatch.setattr(freewater_qc,'diagnostic_multishell',lambda *args: (values,diagnostics))
    b = np.r_[0, np.full(6,700),np.full(6,2000)]
    result = freewater_qc.fit_multishell_checked(np.ones((3,13))*100,b,np.zeros((3,13)))
    assert result['physical'].tolist() == [True,False,False]
    assert result['raw_min_eigenvalue'][1] < 0


def test_multishell_recovers_known_two_compartment_signal():
    rng = np.random.default_rng(20260915)
    directions = rng.normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    vectors = np.vstack([np.zeros((2, 3)), directions, directions])
    bvals = np.r_[0., 0., np.full(30, 700.), np.full(30, 2000.)]
    tensor = np.diag([1.2e-3, .4e-3, .4e-3])
    tissue = np.exp(-bvals * np.einsum('ni,ij,nj->n', vectors, tensor, vectors))
    water = np.exp(-bvals * 3e-3)
    fractions = np.array([.1, .25, .4])
    signal = 1000 * ((1-fractions[:, None])*tissue + fractions[:, None]*water)
    result = freewater_qc.fit_multishell_checked(signal, bvals, vectors.T)
    assert result['physical'].all()
    assert np.allclose(result['f'], fractions, atol=1e-3)
    assert np.allclose(result['tissue_md'], np.trace(tensor)/3, atol=1e-6)
    assert np.all(result['raw_min_eigenvalue'] > 0)
