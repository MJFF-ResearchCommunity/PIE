"""Explicit, unreviewed rigid MRI alignment for side-by-side visual inspection."""
import hashlib
import json
import SimpleITK as sitk

from .images import volume_info


def prepare_comparison(store, baseline, followup):
    if (baseline.subject != followup.subject or baseline.modality != "MRI" or followup.modality != "MRI"
            or not baseline.date or not followup.date or baseline.date >= followup.date):
        raise ValueError("Select an earlier and later MRI of the same participant")
    left, right = store.prepare(baseline), store.prepare(followup)
    key = hashlib.sha256(f"rigid-review-v1|{left['fingerprint']}|{right['fingerprint']}".encode()).hexdigest()[:24]
    with store._lock:
        folder = store.cache / f"comparison-{key}"
        folder.mkdir(parents=True, exist_ok=True)
        output, record_path = folder / "followup_in_baseline.nii.gz", folder / "registration.json"
        if not output.exists() or not record_path.exists():
            def read(p):
                v = next(v for v in p["volumes"] if v["role"] == "primary")
                image = sitk.ReadImage(str(store.assets[v["url"].split("/")[-2]]), sitk.sitkFloat32)
                if image.GetDimension() != 3:
                    raise ValueError("Rigid comparison currently accepts 3D anatomical MRI only")
                return image
            fixed, moving = read(left), read(right)
            initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
            registration = sitk.ImageRegistrationMethod()
            registration.SetNumberOfThreads(4)
            registration.SetMetricAsMattesMutualInformation(50)
            registration.SetMetricSamplingStrategy(registration.RANDOM)
            registration.SetMetricSamplingPercentage(0.02, seed=42)
            registration.SetInterpolator(sitk.sitkLinear)
            registration.SetOptimizerAsRegularStepGradientDescent(2., 0.001, 200)
            registration.SetOptimizerScalesFromPhysicalShift()
            registration.SetShrinkFactorsPerLevel([4, 2, 1])
            registration.SetSmoothingSigmasPerLevel([2, 1, 0])
            registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration.SetInitialTransform(initial, inPlace=False)
            transform = registration.Execute(fixed, moving)
            resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0., sitk.sitkFloat32)
            temporary = folder / "followup.tmp.nii.gz"
            sitk.WriteImage(resampled, str(temporary))
            temporary.replace(output)
            record = {"status": "unreviewed", "method": "6-DOF rigid / Mattes mutual information", "interpolation": "linear",
                      "mapping": "fixed baseline LPS mm → moving follow-up LPS mm", "parameters": list(transform.GetParameters()),
                      "fixed_parameters": list(transform.GetFixedParameters()), "metric": registration.GetMetricValue(),
                      "optimizer_stop": registration.GetOptimizerStopConditionDescription(),
                      "baseline_fingerprint": left["fingerprint"], "followup_fingerprint": right["fingerprint"]}
            record_path.write_text(json.dumps(record))
        record = json.loads(record_path.read_text())
        primary = next(v for v in right["volumes"] if v["role"] == "primary")
        preview = {**right, "fingerprint": key, "geometry": volume_info(output), "regions": [], "meshes": [], "extra": [],
                   "volumes": [{**primary, "url": store.asset(output)}],
                   "scan": {**right["scan"], "space": baseline.space, "reference_id": baseline.id, "registration": "unreviewed",
                            "has_atlas": False, "has_anatomy": False, "qc": "Automatic rigid alignment not reviewed",
                            "provenance": "Follow-up MRI resampled into baseline geometry for visual review only. Six rigid parameters; no scaling, deformable warp, intensity normalization, or disease measurement."}}
        return {"baseline": left, "followup": preview, "registration": record, "fingerprint": key}
