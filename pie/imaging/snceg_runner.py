"""Standalone snceg inference, run by ``nm_native.snceg_mask`` inside the snceg environment (torch 2.0.1, fastMONAI
0.4.0.2); it imports nothing from PIE. Adapted from snceg.py, github.com/lillepeder/snceg (MIT licence, (c) Peder
A. G. Lillebostad): the bug-fixed ``_inference`` with the model's stored orientation and spacing (``--resample``, the
authors' recommended mode), the mask returned on the input grid.

    python snceg_runner.py --input nm_mean.nii.gz --output sn.nii.gz --model-dir <dir with SNceg-0.1.pkl>
"""
import argparse
from pathlib import Path


def main():
    from fastMONAI.vision_all import do_pad_or_crop, load_learner, load_variables, med_img_reader
    from fastMONAI.vision_inference import _do_resize, _to_original_orientation

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", required=True)
    a = ap.parse_args()
    model = Path(a.model_dir)
    learner = load_learner(model / "SNceg-0.1.pkl")
    _, reorder, resample = load_variables(pkl_fn=model / "vars_SNceg-0.1.pkl")
    org_img, input_img, org_size = med_img_reader(a.input, reorder=reorder, resample=resample, only_tensor=False)
    pred, *_ = learner.predict(input_img.data)
    input_img.set_data(do_pad_or_crop(pred.float(), input_img.shape[1:], padding_mode=0, mask_name=None))
    input_img = _do_resize(input_img, org_size, image_interpolation="nearest")
    org_img.set_data(_to_original_orientation(input_img.as_sitk(), "".join(org_img.orientation)))
    org_img.save(a.output)


if __name__ == "__main__":
    main()
