"""Standard WLS tensor measurements with explicit pre-clipping physicality QC.

This module does not estimate single-shell free water. Fit parameters and residuals
are retained even when a tensor fails the numerical guard.
"""
import numpy as np


def tensor_measurements(signal,bvals,bvecs):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import design_matrix,wls_fit_tensor,from_lower_triangular,fractional_anisotropy
    signal=np.asarray(signal,float);b=np.asarray(bvals,float);v=np.asarray(bvecs,float)
    if signal.ndim!=2 or signal.shape[1]!=len(b) or v.shape!=(3,len(b)):raise ValueError('Signal/gradient dimensions mismatch')
    if not (b<=50).any() or (b>50).sum()<12:raise ValueError('Own b0 and at least 12 diffusion volumes required')
    g=gradient_table(b,bvecs=v.T,b0_threshold=50);design=design_matrix(g)
    finite=np.isfinite(signal).all(axis=1);positive=(signal>0).all(axis=1)
    # Log-WLS needs positive input. A floor avoids solver crashes, but the affected
    # voxel is never silently accepted as an ordinary positive-signal fit.
    y=np.maximum(np.nan_to_num(signal,nan=1e-6,posinf=1e-6,neginf=1e-6),1e-6)
    params,_=wls_fit_tensor(design,y,return_lower_triangular=True)
    evals=np.linalg.eigvalsh(from_lower_triangular(params[...,:6]))[:,::-1]
    accepted=finite&positive&np.isfinite(params).all(axis=1)&(evals.min(axis=1)>=-1e-12)
    safe=np.maximum(evals,0)
    md=safe.mean(axis=1);fa=fractional_anisotropy(safe);ad=safe[:,0];rd=safe[:,1:].mean(axis=1)
    pred=np.exp(np.clip(params@design.T,-50,50));s0=y[:,b<=50].mean(axis=1)
    rmse=np.sqrt(np.mean((pred-y)**2,axis=1))/s0
    accepted &= np.isfinite(rmse)&(s0>0)
    return {'md':np.where(accepted,md,np.nan),'fa':np.where(accepted,fa,np.nan),
            'ad':np.where(accepted,ad,np.nan),'rd':np.where(accepted,rd,np.nan),
            'accepted':accepted,'relative_rmse':rmse,'raw_min_eigenvalue':evals.min(axis=1),
            'nonpositive_signal':~positive,'nonfinite_signal':~finite}
