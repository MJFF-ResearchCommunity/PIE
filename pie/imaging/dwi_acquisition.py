"""Opt-in acquisition-aware DWI preparation and solver instrumentation.

Development path, not the default cohort pipeline. Only origin differences are
regridded: axes, spacing, shape, TE/TR and PE/readout must agree within a group.
No estimated intensity normalization is applied. Eddy metadata are never guessed.
"""
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from . import dwi

# Diffusivity is in mm^2/s. This only tolerates numerical eigensolver roundoff,
# not a biologically meaningful negative diffusivity or a singular-model remedy.
TENSOR_EIGENVALUE_TOLERANCE = 1e-12


def acceptable_tensor_fit(diag,fw,error=''):
    return bool(not error and diag['status'] in [1,2,3,4]
                and np.isfinite(diag['relative_rmse'])
                and np.isfinite(diag['raw_tensor_min_eigenvalue'])
                and diag['raw_tensor_min_eigenvalue']>=-TENSOR_EIGENVALUE_TOLERANCE
                and np.isfinite(fw) and 0<=fw<=1)


def positive_definite_initial_tensor(tensor_elements,floor=1e-6):
    """Development-only repair of a singular WLS *initialization*, in mm^2/s.

    The NLS solution is not floored afterwards. This is not a validated change
    to the estimator and is never applied by the default cohort path.
    """
    from dipy.reconst.dti import from_lower_triangular,lower_triangular
    if not np.isfinite(floor) or floor<=0:raise ValueError('Positive finite initialization floor required')
    matrix=from_lower_triangular(np.asarray(tensor_elements,float))
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite initial tensor')
    evals,evecs=np.linalg.eigh(matrix)
    if evals.min()>=floor:return np.asarray(tensor_elements,float)
    return lower_triangular((evecs*np.maximum(evals,floor))@evecs.T)


def load_runs(files):
    rows=[]
    for nii,bval,bvec,sidecar in files:
        img=nib.load(nii);b=np.loadtxt(bval).ravel();v=np.loadtxt(bvec).reshape(3,-1)
        if img.ndim!=4 or len(b)!=img.shape[3] or v.shape[1]!=len(b):
            raise ValueError(f'Invalid DWI volume/gradient dimensions: {nii}')
        if not np.isfinite(b).all() or not np.isfinite(v).all():raise ValueError('Nonfinite gradients')
        if (b>50).any() and not np.allclose(np.linalg.norm(v[:,b>50],axis=0),1,atol=.01):
            raise ValueError('Diffusion gradients are not unit length')
        meta=json.loads(Path(sidecar).read_text())
        rows.append({'image':img,'bvals':b,'bvecs':v,'meta':meta,'path':str(nii)})
    return rows


def acquisition_key(run):
    img,m=run['image'],run['meta']
    axes=img.affine[:3,:3];u=axes/np.linalg.norm(axes,axis=0)
    if not np.allclose(u.T@u,np.eye(3),atol=1e-5):raise ValueError('Sheared acquisition grid')
    return (img.shape[:3],tuple(np.round(axes,5).ravel()),dwi.acquisition_metadata_key(m))


def choose_group(runs, minimum_directions=12):
    groups={}
    for run in runs:
        if (run['bvals']>50).sum()>=minimum_directions:
            if not (run['bvals']<=50).any():raise ValueError('Acquisition has no own b0')
            groups.setdefault(acquisition_key(run),[]).append(run)
    if not groups:raise ValueError('No diffusion-weighted acquisitions')
    return max(groups.values(),key=lambda group:sum((r['bvals']>50).sum() for r in group))


def regrid_volume(array,source_affine,reference_shape,reference_affine):
    """Same physical points on a new origin-aligned grid; not estimated head-motion correction."""
    import SimpleITK as sitk
    if not np.allclose(source_affine[:3,:3],reference_affine[:3,:3],atol=1e-5,rtol=0):
        raise ValueError('Changed axes/spacing require explicit PE and gradient-frame handling')
    if array.shape==tuple(reference_shape) and np.allclose(source_affine,reference_affine,atol=1e-5,rtol=0):
        return np.asarray(array,dtype=np.float32)
    source=dwi._sitk_native(array,source_affine)
    reference=dwi._sitk_native(np.zeros(reference_shape,np.float32),reference_affine)
    result=sitk.Resample(source,reference,sitk.Transform(3,sitk.sitkIdentity),sitk.sitkLinear,0.)
    return sitk.GetArrayFromImage(result).transpose(2,1,0)


def prepare_group(runs):
    group=choose_group(runs);ref=group[0]['image'];data=[];b=[];v=[];records=[];offset=0
    for k,run in enumerate(group):
        img=run['image'];arr=np.asarray(img.dataobj,dtype=np.float32)
        shifted=not np.allclose(img.affine,ref.affine,atol=1e-5,rtol=0)
        if shifted:
            arr=np.stack([regrid_volume(arr[...,i],img.affine,ref.shape[:3],ref.affine) for i in range(arr.shape[3])],axis=-1)
        data.append(arr);b.append(run['bvals']);v.append(run['bvecs'])
        records.append({'run':k,'source':run['path'],'start':offset,'stop':offset+len(run['bvals']),
            'source_affine':img.affine.tolist(),'reference_affine':ref.affine.tolist(),'origin_regridded':shifted,
            'origin_shift_mm':float(np.linalg.norm(img.affine[:3,3]-ref.affine[:3,3])),
            'b0_indices':(offset+np.flatnonzero(run['bvals']<=50)).tolist(),'metadata':run['meta']})
        offset+=len(run['bvals'])
    # Axes do not change and there is no estimated rigid rotation in this step,
    # so FSL bvecs and PE axes are unchanged. Subsequent motion correction rotates bvecs.
    b=np.concatenate(b)
    return {'data':np.concatenate(data,axis=-1),'bvals':b,'bvecs':np.concatenate(v,axis=1),
        'affine':ref.affine,'meta':group[0]['meta'],'n_runs':len(group),'source_nifti':[r['path'] for r in group],
        'shells':sorted(set(int(x) for x in np.round(b[b>50]/100).astype(int)*100)), 'run_records':records}


def pe_row(meta):
    pe=meta.get('PhaseEncodingDirection');readout=meta.get('TotalReadoutTime')
    if pe not in ['i','i-','j','j-','k','k-'] or readout is None or not np.isfinite(readout) or readout<=0:
        raise ValueError('Verified PhaseEncodingDirection and positive TotalReadoutTime required; no guessed metadata')
    vec=np.zeros(3);vec['ijk'.index(pe[0])]=-1 if pe.endswith('-') else 1
    return np.r_[vec,float(readout)]


def diagnostic_multishell(signal,bvals,bvecs,cholesky=False,cholesky_init_floor=None):
    """Instrument unchanged DIPY NLS one voxel at a time; expose MINPACK status.

    The temporary scipy wrapper is local to this single-threaded fitting process;
    do not invoke concurrently with other Python optimizer work in the same process.
    """
    from unittest.mock import patch
    from dipy.core.gradients import gradient_table
    from dipy.reconst.fwdti import FreeWaterTensorModel
    from dipy.reconst import fwdti
    from dipy.reconst.dti import from_lower_triangular
    import warnings
    from contextlib import ExitStack
    if cholesky_init_floor is not None and not cholesky:raise ValueError('Initialization floor requires Cholesky fitting')
    signal=np.asarray(signal,float);gt=gradient_table(bvals,bvecs=bvecs.T,b0_threshold=50)
    model=FreeWaterTensorModel(gt,cholesky=cholesky);original=fwdti.opt.leastsq
    original_cholesky=fwdti.lower_triangular_to_cholesky
    values=[];diagnostics=[]
    for i,y in enumerate(signal):
        calls=[];initializations=[]
        def initialize(tensor):
            adjusted=positive_definite_initial_tensor(tensor,cholesky_init_floor)
            initializations.append(not np.array_equal(adjusted,tensor))
            return original_cholesky(adjusted)
        def capture(*args,**kwargs):
            kwargs['full_output']=True
            params,cov,info,message,status=original(*args,**kwargs)
            tensor=fwdti.cholesky_to_lower_triangular(params[:6]) if cholesky else params[:6]
            minimum=float(np.linalg.eigvalsh(from_lower_triangular(tensor)).min()) if np.isfinite(tensor).all() else np.nan
            calls.append({'status':int(status),'nfev':int(info['nfev']),'message':message,
                          'relative_rmse':float(np.sqrt(np.mean(info['fvec']**2))/max(y[np.asarray(bvals)<=50].mean(),1e-12)),
                          'raw_tensor_min_eigenvalue':minimum})
            return params,status
        error='';fit=None
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always',RuntimeWarning)
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(fwdti.opt,'leastsq',capture))
                    if cholesky_init_floor is not None:
                        stack.enter_context(patch.object(fwdti,'lower_triangular_to_cholesky',initialize))
                    fit=model.fit(y)
            except (np.linalg.LinAlgError,ValueError,FloatingPointError) as exc:
                error=f'{type(exc).__name__}: {exc}'
        diag=calls[0] if calls else {'status':0,'nfev':0,'message':'DIPY WLS threshold branch; no NLS solve',
                                   'relative_rmse':np.nan,'raw_tensor_min_eigenvalue':np.nan}
        if error and not calls:diag['message']='Fit failed before a completed NLS solve'
        fw=float(fit.f) if fit is not None else np.nan
        valid=bool(calls and acceptable_tensor_fit(diag,fw,error))
        diagnostics.append(dict(diag,voxel=i,accepted=valid,error=error,
                                initialization_adjusted=any(initializations),
                                warnings=' | '.join(sorted({str(w.message) for w in recorded}))))
        values.append({'fw':fw,'tissue_md':float(fit.md) if fit is not None else np.nan,
                       'tissue_fa':float(fit.fa) if fit is not None else np.nan,
                       'fw_accepted':fw if valid else np.nan})
    return values,diagnostics
