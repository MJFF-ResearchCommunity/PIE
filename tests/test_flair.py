"""Unit test for the FLAIR white-matter-hyperintensity threshold on a phantom."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging import flair


def test_wmh_threshold_recovers_lesions_and_splits_periventricular():
    shape = (40, 60, 60)
    aseg = np.zeros(shape, dtype=np.int32)
    aseg[5:35, 5:55, 5:55] = 2                 # white matter block
    aseg[15:25, 25:35, 25:35] = 4              # a lateral ventricle in the middle
    rng = np.random.default_rng(0)
    img = np.where(aseg > 0, 100.0, 0.0) + rng.normal(0, 3, shape)
    img[aseg == 4] = 20.0                      # CSF dark on FLAIR
    img[18:22, 36:40, 28:32] = 160.0           # periventricular lesion (touches the ventricle, 64 mm3)
    img[8:11, 45:50, 45:50] = 150.0            # deep lesion (75 mm3, ~20 mm from the ventricle)
    img[30, 50, 50] = 200.0                    # single bright voxel: below the minimum size, must be dropped
    les, out = flair.wmh(img.astype(np.float32), aseg, vox_mm=1.0)
    assert 120 <= out["wmh_mm3"] <= 150, out["wmh_mm3"]
    assert 55 <= out["wmh_pv_mm3"] <= 70 and 65 <= out["wmh_deep_mm3"] <= 80
    assert out["wmh_n_lesions"] == 2
    assert not les[30, 50, 50]
    assert abs(out["flair_wm_median"] - 100) < 2 and out["wmh_threshold"] > 105
