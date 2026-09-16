"""
embed.py — fixed-length T1 embeddings from pretrained open-weight brain MRI models (BrainIAC, 3D-Neuro-SimCLR, SFCN).

The downstream question is whether a generic image representation carries information the FastSurfer region
features miss. Every backend reads the FastSurfer `orig_nu.mgz` (N4-corrected, 1 mm, 256^3) masked with `mask.mgz`,
resampled linearly into MNI space with the package's cached affine (`pie.imaging.dwi.register_t1_to_mni`, fitted on
the 2 mm nilearn MNI152NLin2009cAsym brain; ~6 s when not cached) onto the 1 mm grid each model was trained on, then
normalised the way the authors' dataset code does:

    brainiac  Tak et al. 2026 (Nat Neurosci), MONAI ViT-B/16 on 96^3, 768-d. Authors: rigid to their head template
              (170 x 206 x 162 mm box, LAS), N4, HD-BET, `Resized` to 96^3 (anisotropic trilinear), z-score of nonzero
              voxels. Here: same box and voxel order, affine instead of rigid, FastSurfer mask instead of HD-BET.
              The embedding is token 0 of the final layer as in their `ViTBackboneNet` — the checkpoint has no
              CLS token (216 = 6^3 patch positions), so this is the first patch token, not a true CLS.
    simclr    Kaczmarek et al. 2025 (arXiv 2509.10620), 3D ResNet-18 (MONAI), 512-d global-average-pooled features.
              Authors: TurboPrep (N4, SynthStrip, rigid to MNI152 ICBM 2009c sym 193 x 229 x 193, WhiteStripe),
              array transposed to (z, y, x) and cropped to [10:160, 19:211, 1:] -> 150 x 192 x 192, z-scored inside
              the mask. Here: same box (x -95..96, y -113..78, z -68..81 mm), affine instead of rigid; WhiteStripe
              is a linear intensity map, so the masked z-score makes it redundant.
    sfcn      Peng et al. 2021 (MedIA) brain-age SFCN trained on UK Biobank `T1_brain_to_MNI` (FLIRT, MNI152 1 mm
              182 x 218 x 182, LAS): `x / x.mean()` over the full grid, then centre crop to 160 x 192 x 160 (their
              `examples.ipynb`; the order matters — with this convention the first BatchNorm's running statistics
              match our inputs at scale 1.00, dividing by the mean of the crop instead leaves the softmax flat).
              40-bin softmax over ages 42-82: returns the 64 penultimate channels (1x1 conv + average pool) and the
              expected age `brainage_sfcn`. PPMI ages below 42 fall outside the training range. MNI152NLin6 (FSL)
              vs NLin2009c differ by ~1-2 mm, ignored here.

Weights live under `third_party/weights/<backend>/` (see WEIGHTS.md there; `PIE_WEIGHTS_DIR` overrides the root); a backend whose weights are missing is
skipped with a message. Licenses: BrainIAC research-only (Mass General Brigham); 3D-Neuro-SimCLR MIT; SFCN MIT
(LICENSE file present in the UKBiobank_deep_pretrain repo despite earlier reports of none).

    from pie.imaging.embed import GRID, to_mni, embed_sfcn, load_net
    vol = to_mni("Imaging/derived/fastsurfer", "I000001", *GRID["sfcn"])      # (182, 218, 182) RAS, zeros outside brain
    v = embed_sfcn(vol, load_net("sfcn", "cuda"))                             # 64 features + brain age

    python -m pie.imaging.embed --fastsurfer-dir Imaging/derived/fastsurfer --ids-csv dataset.csv --out emb.csv \\
        --backends brainiac simclr sfcn --device cuda
"""

import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .dwi import _brain, mni_cache_path, register_t1_to_mni

WEIGHTS_DIR = Path(os.environ.get("PIE_WEIGHTS_DIR") or Path(__file__).resolve().parents[2] / "third_party" / "weights")
WEIGHTS = {"brainiac": WEIGHTS_DIR / "brainiac" / "BrainIAC.ckpt",
           "simclr": WEIGHTS_DIR / "simclr" / "simclr_3d_brain_foundation.tar",
           "sfcn": WEIGHTS_DIR / "sfcn" / "run_20190719_00_epoch_best_mae.p"}
# 1 mm MNI grid per backend: (shape in voxels, RAS coordinate of voxel (0, 0, 0)); arrays come out in (x, y, z) RAS order
GRID = {"brainiac": ((170, 206, 162), (-85.0, -119.0, -69.0)),
        "simclr": ((192, 192, 150), (-95.0, -113.0, -68.0)),
        "sfcn": ((182, 218, 182), (-91.0, -126.0, -72.0))}       # FSL MNI152 box; embed_sfcn crops to 160 x 192 x 160
DIM = {"brainiac": 768, "simclr": 512, "sfcn": 64}
SFCN_AGES = np.arange(42, 82) + 0.5


def to_mni(fastsurfer_dir, image_id, shape, origin):
    """Brain-masked `orig_nu` resampled (linear) with the cached T1->MNI affine onto a 1 mm RAS grid of ``shape`` voxels
    whose first voxel sits at RAS ``origin`` (mm). Returns float32 (x, y, z) with zeros outside the brain."""
    import SimpleITK as sitk

    sub = Path(fastsurfer_dir) / image_id
    t1, mask = nib.load(sub / "mri" / "orig_nu.mgz"), nib.load(sub / "mri" / "mask.mgz")
    tx, _ = register_t1_to_mni(t1, mask, cache_path=mni_cache_path(sub))
    ref = sitk.Image([int(s) for s in shape], sitk.sitkFloat32)
    ref.SetOrigin((-origin[0], -origin[1], origin[2]))                      # RAS -> LPS, as in dwi._sitk_from_nib
    ref.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    out = sitk.Resample(_brain(t1, mask, mm=1.0), ref, tx, sitk.sitkLinear, 0.0)
    return np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0)).astype(np.float32)


def brainiac_net():
    from monai.networks.nets import ViT
    return ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16), hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12)


def simclr_net():
    from monai.networks.nets import resnet18
    return resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, bias_downsample=True)


def sfcn_net(channels=(32, 64, 128, 256, 256, 64), n_bins=40):
    """SFCN with the authors' module names (feature_extractor.conv_i, classifier.conv_6) so their state dict loads."""
    import torch.nn as nn
    fe, c_in = nn.Sequential(), 1
    for i, c in enumerate(channels):
        last = i == len(channels) - 1
        layers = [nn.Conv3d(c_in, c, 1 if last else 3, padding=0 if last else 1), nn.BatchNorm3d(c)]
        fe.add_module(f"conv_{i}", nn.Sequential(*layers, *([] if last else [nn.MaxPool3d(2)]), nn.ReLU()))
        c_in = c
    clf = nn.Sequential(OrderedDict(average_pool=nn.AvgPool3d((5, 6, 5)), dropout=nn.Dropout(0.5), conv_6=nn.Conv3d(c_in, n_bins, 1)))
    return nn.Sequential(OrderedDict(feature_extractor=fe, classifier=clf))


def load_net(name, device="cuda"):
    """Architecture + pretrained weights in eval mode on ``device``; raises FileNotFoundError when the weights are absent."""
    import torch
    path = WEIGHTS[name]
    if not path.exists():
        raise FileNotFoundError(f"{name}: weights not found at {path} (see third_party/weights/WEIGHTS.md)")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if name == "brainiac":
        net = brainiac_net()
        sd = {k[len("backbone."):]: v for k, v in ck["state_dict"].items() if k.startswith("backbone.")}
        missing, unexpected = net.load_state_dict(sd, strict=False)     # MONAI 1.5 registers unused cross-attention modules
        assert not unexpected and all("cross_attn" in k for k in missing), (missing, unexpected)
    elif name == "simclr":
        net = simclr_net()
        sd = {k.replace("module.", "", 1): v for k, v in ck["model_state_dict"].items()}
        net.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
    else:
        net = sfcn_net()
        net.load_state_dict({k.replace("module.", "", 1): v for k, v in ck.items()}, strict=True)
    return net.to(device).eval()


def _forward(net, x, fn):
    import torch
    device = next(net.parameters()).device
    with torch.no_grad():
        return fn(net, torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None].to(device)).float().cpu().numpy()


def embed_brainiac(volume, net):
    """``volume``: (170, 206, 162) RAS from GRID['brainiac']. LAS voxel order, trilinear resize to 96^3, z-score of the
    nonzero voxels (MONAI Resized + NormalizeIntensityd(nonzero=True)); returns token 0 of the last layer (768)."""
    import torch
    import torch.nn.functional as F

    def fn(net, x):
        x = F.interpolate(x, size=(96, 96, 96), mode="trilinear", align_corners=False)
        nz = x != 0
        v = x[nz]
        x = torch.where(nz, (x - v.mean()) / (v.std(unbiased=False) + 1e-8), x)
        return net(x)[0][:, 0]

    return _forward(net, volume[::-1], fn)[0]


def embed_simclr(volume, net):
    """``volume``: (192, 192, 150) RAS from GRID['simclr']. Transposed to (z, y, x) = 150 x 192 x 192, z-scored inside
    the brain (zeros outside) as the authors' ``standardize``; returns the 512 pooled ResNet-18 features."""
    x = np.transpose(volume, (2, 1, 0))
    m = x != 0
    v = x[m]
    x = np.where(m, (x - v.mean()) / (v.std() + 1e-6), 0.0)
    return _forward(net, x, lambda net, t: net(t))[0]


def embed_sfcn(volume, net):
    """``volume``: (182, 218, 182) RAS from GRID['sfcn']. LAS voxel order, ``x / x.mean()`` over the full grid, then the
    authors' centre crop to 160 x 192 x 160. Returns 65 values: the 64 average-pooled penultimate channels, then the
    expected age over the 40 one-year bins."""
    import torch

    def fn(net, x):
        f = net.feature_extractor(x)
        p = torch.softmax(net.classifier(f).flatten(1), dim=1)
        return torch.cat([f.mean(dim=(2, 3, 4)), p @ torch.as_tensor(SFCN_AGES, dtype=p.dtype, device=p.device)[:, None]], dim=1)

    x = volume[::-1] / (volume.mean() + 1e-8)
    return _forward(net, x[11:171, 13:205, 11:171], fn)[0]


EMBED = {"brainiac": embed_brainiac, "simclr": embed_simclr, "sfcn": embed_sfcn}


def columns(backend):
    cols = [f"emb_{backend}_{k}" for k in range(DIM[backend])]
    return cols + ["brainage_sfcn"] if backend == "sfcn" else cols


def _mni_or_error(args):
    """Worker: the MNI-resampled volume for one subject and backend, or the exception (so the main loop can log and continue)."""
    fastsurfer_dir, iid, backend = args
    try:
        return to_mni(fastsurfer_dir, iid, *GRID[backend])
    except Exception as e:      # noqa: BLE001
        return e


def run(fastsurfer_dir, image_ids, backends, out_csv, device="cuda", log=print, workers=4):
    """One backend at a time (one network on the GPU): rows of IMAGEID + that backend's columns appended to
    ``<out stem>_<backend>.csv`` (ids already present are skipped), then the per-backend files are joined on IMAGEID into
    ``out_csv``. Backends without weights are dropped with a message. Returns the number of rows written."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    out_csv = Path(out_csv)
    parts, n = [], 0
    for b in backends:
        try:
            net = load_net(b, device)
        except FileNotFoundError as e:
            log(f"skipping backend: {e}")
            continue
        part = out_csv.with_name(f"{out_csv.stem}_{b}.csv")
        done = set(pd.read_csv(part, usecols=["IMAGEID"], dtype=str)["IMAGEID"]) if part.exists() else set()
        todo = [i for i in map(str, image_ids) if i not in done]
        log(f"{b}: {len(todo)} subjects to embed ({len(done)} already in {part})", flush=True)
        t0, k = time.time(), 0
        with ProcessPoolExecutor(workers) as pool:      # MNI resampling on the CPU in parallel, inference on the GPU in this process
            for iid, vol in zip(todo, pool.map(_mni_or_error, [(fastsurfer_dir, iid, b) for iid in todo], chunksize=2)):
                try:
                    if isinstance(vol, Exception):
                        raise vol
                    row = {"IMAGEID": iid, **dict(zip(columns(b), EMBED[b](vol, net)))}
                except Exception as e:                  # missing FastSurfer output, unreadable volume: report and go on
                    log(f"{b} {iid}: failed ({str(e)[:120]})", flush=True)
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    continue
                pd.DataFrame([row]).to_csv(part, mode="a", header=not part.exists(), index=False)
                n += 1
                k += 1
                if k % 50 == 0:
                    log(f"{b}: {k}/{len(todo)} ({(time.time() - t0) / k:.1f} s/subject)", flush=True)
        parts.append(part)
        del net
        torch.cuda.empty_cache()
    if parts:
        merged = None
        for part in parts:
            d = pd.read_csv(part, dtype={"IMAGEID": str}).drop_duplicates("IMAGEID")
            merged = d if merged is None else merged.merge(d, on="IMAGEID", how="outer")
        merged.to_csv(out_csv, index=False)
        log(f"merged {len(merged)} subjects x {merged.shape[1] - 1} columns -> {out_csv}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fastsurfer-dir", required=True)
    ap.add_argument("--ids-csv", required=True, help="CSV with an IMAGEID column (one row per subject)")
    ap.add_argument("--out", required=True, help="output CSV (appended to; already-embedded ids are skipped)")
    ap.add_argument("--backends", nargs="+", default=list(EMBED), choices=list(EMBED))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4, help="CPU processes for the MNI resampling")
    ap.add_argument("--limit", type=int, default=None, help="embed at most this many subjects (after resume filtering)")
    a = ap.parse_args(argv)
    ids = pd.read_csv(a.ids_csv, usecols=["IMAGEID"], dtype=str)["IMAGEID"].dropna().drop_duplicates().tolist()
    if a.limit:
        done = set(pd.read_csv(a.out, usecols=["IMAGEID"], dtype=str)["IMAGEID"]) if Path(a.out).exists() else set()
        ids = [i for i in ids if i not in done][:a.limit]
    run(a.fastsurfer_dir, ids, a.backends, a.out, a.device, workers=a.workers)


if __name__ == "__main__":
    main()
