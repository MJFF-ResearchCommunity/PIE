"""
cnn.py — 3D CNN baseline on FastSurfer-conformed T1 volumes (SFCN-style), with grouped out-of-fold predictions.

The reviewer question this answers: "could a network reading the whole image find a pattern the region features
miss?" Volumes are the FastSurfer `orig_nu.mgz` (bias-corrected, 1 mm, 256^3, LIA) masked with `mask.mgz`, z-scored
inside the brain, cropped to a fixed box around the brain centroid and mean-pooled to 2 mm (88 x 96 x 88 voxels).
The network is the SFCN of Peng et al. (2021, brain age) with a sigmoid head: five conv-BN-ReLU-maxpool blocks
(32-64-128-256-256 channels), a 1x1 conv, global average pooling, dropout, one logit. Training uses AdamW, weighted
binary cross-entropy, left-right flips and random shifts, mixed precision, and early stopping on an inner validation
split; predictions are out-of-fold over patient-grouped stratified folds, so they can be compared with the tabular
models on the same subjects.

    from pie.imaging.cnn import cache_volumes, cross_validate
    X = cache_volumes("Imaging/derived/fastsurfer", image_ids, "volumes.npy")     # (n, 88, 96, 88) float16 memmap
    oof = cross_validate(X, y, groups, n_splits=5, epochs=30, device="cuda")

    python -m pie.imaging.cnn --labels labels.csv --fastsurfer-dir Imaging/derived/fastsurfer --cache vol.npy --out oof.csv

Transfer learning: ``--pretrained`` fine-tunes the SFCN brain-age weights of Peng et al. (UK Biobank, 1 mm MNI input;
pie.imaging.embed holds the weights and the MNI resampling) with a fresh one-logit head, on 160 x 192 x 160 MNI volumes
(`load_volume_mni`, 9.8 MB each as float16: keep that cache on a large disk). A from-scratch 3D CNN on ~700 subjects is a
weak reviewer baseline; a pretrained backbone is the fair one.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SHAPE = (88, 96, 88)          # after 2x pooling of a 176 x 192 x 176 mm box (the brain bounding box is ~142 x 148 x 166)


def load_volume(fastsurfer_dir, image_id, shape=SHAPE):
    """Masked, z-scored, centred and 2 mm-pooled T1 as float16 (zeros outside the brain)."""
    mri = Path(fastsurfer_dir) / image_id / "mri"
    img = nib.load(mri / "orig_nu.mgz").get_fdata(dtype=np.float32)
    mask = nib.load(mri / "mask.mgz").get_fdata() > 0
    if mask.sum() < 1000:
        raise ValueError(f"{image_id}: empty brain mask")
    img = np.where(mask, img, 0.0)
    v = img[mask]
    img[mask] = (v - v.mean()) / (v.std() + 1e-6)
    c = np.array(np.nonzero(mask)).mean(axis=1).round().astype(int)
    box = tuple(2 * s for s in shape)
    out = np.zeros(box, np.float32)
    src, dst = [], []
    for ax in range(3):                                  # box around the centroid, clipped at the volume edge
        lo = int(c[ax]) - box[ax] // 2
        s0, s1 = max(lo, 0), min(lo + box[ax], img.shape[ax])
        src.append(slice(s0, s1))
        dst.append(slice(s0 - lo, s1 - lo))
    out[tuple(dst)] = img[tuple(src)]
    pooled = out.reshape(shape[0], 2, shape[1], 2, shape[2], 2).mean(axis=(1, 3, 5))
    return pooled.astype(np.float16)


def cache_volumes(fastsurfer_dir, image_ids, cache_path, shape=SHAPE, loader=None):
    """Load every volume once into a float16 memmap (n x shape); reuse the cache if it matches the id list.
    ``loader(fastsurfer_dir, image_id)`` defaults to ``load_volume`` (2 mm conformed); use ``load_volume_mni`` with ``MNI_SHAPE``."""
    loader = loader or (lambda d, i: load_volume(d, i, shape))
    cache, ids = Path(cache_path), Path(str(cache_path) + ".ids")
    if cache.exists() and ids.exists() and ids.read_text().split() == list(map(str, image_ids)):
        return np.load(cache, mmap_mode="r")
    X = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16, shape=(len(image_ids), *shape))
    for i, iid in enumerate(image_ids):
        X[i] = loader(fastsurfer_dir, iid)
        if i % 100 == 0:
            print(f"cached {i}/{len(image_ids)}", flush=True)
    X.flush()
    ids.write_text("\n".join(map(str, image_ids)))
    return np.load(cache, mmap_mode="r")


def load_volume_mni(fastsurfer_dir, image_id):
    """SFCN input: brain-masked T1 on the FSL MNI152 1 mm grid, LAS voxel order, divided by the full-grid mean, centre-cropped
    to 160 x 192 x 160 (the authors' convention, as in pie.imaging.embed.embed_sfcn); float16."""
    from .embed import GRID, to_mni
    vol = to_mni(fastsurfer_dir, image_id, *GRID["sfcn"])
    x = vol[::-1] / (vol.mean() + 1e-8)
    return np.ascontiguousarray(x[11:171, 13:205, 11:171]).astype(np.float16)


MNI_SHAPE = (160, 192, 160)


def sfcn_pretrained(weights="default"):
    """Peng et al.'s SFCN with the brain-age head replaced by one logit; ``weights=None`` gives the architecture untrained."""
    import torch
    import torch.nn as nn
    from .embed import WEIGHTS, sfcn_net
    net = sfcn_net()
    if weights is not None:
        path = WEIGHTS["sfcn"] if weights == "default" else weights
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict({k.replace("module.", "", 1): v for k, v in ck.items()}, strict=True)
    net.classifier.conv_6 = nn.Conv3d(64, 1, 1)
    return nn.Sequential(net, nn.Flatten())


def sfcn(channels=(32, 64, 128, 256, 256), dropout=0.5):
    import torch.nn as nn
    layers, c_in = [], 1
    for c in channels:
        layers += [nn.Conv3d(c_in, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True), nn.MaxPool3d(2)]
        c_in = c
    layers += [nn.Conv3d(c_in, 64, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d(1), nn.Flatten(),
               nn.Dropout(dropout), nn.Linear(64, 1)]
    return nn.Sequential(*layers)


def _augment(x, rng):
    """Random left-right flip (axis 1 of the batch tensor = L-R in LIA) and a shift of up to 4 voxels per axis."""
    import torch
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[2])
    shifts = [int(s) for s in rng.integers(-4, 5, size=3)]
    return torch.roll(x, shifts=shifts, dims=(2, 3, 4))


def _predict(model, X, device, batch=8):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.asarray(X[i:i + batch], np.float32))[:, None].to(device)
            out.append(torch.sigmoid(model(xb).float()).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def train_fold(X, y, tr, va, te, epochs=30, batch=8, lr=1e-3, device="cuda", seed=0, patience=8, model_fn=sfcn):
    """Fit on tr, early-stop on va (AUROC), return probabilities for te. ``model_fn()`` builds the network."""
    import torch
    from sklearn.metrics import roc_auc_score
    # Keep label and image order identical, including for callers supplying shuffled indices.
    tr, va, te = (np.asarray(idx, dtype=int) for idx in (tr, va, te))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model_fn().to(device)
    pos = y[tr].mean()
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos) / max(pos, 1e-3)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb = torch.from_numpy(np.asarray(X[np.sort(idx)], np.float32))[:, None].to(device)
            yb = torch.from_numpy(y[np.sort(idx)].astype(np.float32)).to(device)
            xb = _augment(xb, rng)
            with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
                loss = crit(model(xb).squeeze(1).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        auc = roc_auc_score(y[va], _predict(model, X[va], device)) if len(np.unique(y[va])) > 1 else 0.5
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return _predict(model, X[te], device), best, ep + 1


def cross_validate(X, y, groups, n_splits=5, repeats=1, epochs=30, device="cuda", seed=0, val_frac=0.15, log=print, model_fn=sfcn, batch=8, lr=1e-3, patience=8):
    """Patient-grouped stratified out-of-fold probabilities (averaged over repeats); inner split for early stopping."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y).astype(int)
    oof = np.zeros((repeats, len(y)))
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr_all, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            inner = StratifiedGroupKFold(n_splits=int(round(1 / val_frac)), shuffle=True, random_state=seed + r)
            tr, va = next(inner.split(np.zeros(len(tr_all)), y[tr_all], np.asarray(groups)[tr_all]))
            tr, va = tr_all[tr], tr_all[va]
            p, val_auc, n_ep = train_fold(X, y, tr, va, te, epochs=epochs, batch=batch, lr=lr, device=device, seed=seed + 100 * r + k, patience=patience, model_fn=model_fn)
            oof[r, te] = p
            log(f"repeat {r} fold {k}: n_train {len(tr)} val AUROC {val_auc:.3f} after {n_ep} epochs")
    return oof.mean(axis=0)


def cross_validate_fusion(X, y, groups, demographics, n_splits=5, repeats=1, inner_splits=3,
                          val_frac=0.15, seed=0, log=print, **train_kwargs):
    """Nested CNN + demographic fusion. Outer test subjects are excluded from every inner CNN fit.

    Each outer fold trains inner CNNs to create meta-training scores, plus one CNN for outer-test scores.
    This costs (inner_splits + 1) times the CNN fits of plain CV. Never cross-validate a meta-model over
    globally computed OOF CNN scores: its training features can depend on its test participants' labels.
    Returns (CNN, demographics, fusion) OOF probabilities averaged over repeats.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y, groups, demo = np.asarray(y, dtype=int), np.asarray(groups), np.asarray(demographics, dtype=float)
    result = np.zeros((3, repeats, len(y)))

    def predict_fold(pool, test, fold_seed):
        splitter = StratifiedGroupKFold(n_splits=int(round(1 / val_frac)), shuffle=True, random_state=fold_seed)
        train, valid = next(splitter.split(np.zeros(len(pool)), y[pool], groups[pool]))
        p, _, _ = train_fold(X, y, pool[train], pool[valid], test, seed=fold_seed, **train_kwargs)
        return p

    def logistic():
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    for rep in range(repeats):
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for k, (train, test) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
            fold_seed = seed + rep * 100 + k
            inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=fold_seed)
            meta_train = np.full(len(train), np.nan)
            for j, (fit, valid) in enumerate(inner.split(np.zeros(len(train)), y[train], groups[train])):
                meta_train[valid] = predict_fold(train[fit], train[valid], fold_seed * 10 + j)
            p_cnn = predict_fold(train, test, fold_seed)
            fusion = logistic().fit(np.column_stack([logit(meta_train), demo[train]]), y[train])
            result[0, rep, test] = p_cnn
            result[1, rep, test] = logistic().fit(demo[train], y[train]).predict_proba(demo[test])[:, 1]
            result[2, rep, test] = fusion.predict_proba(np.column_stack([logit(p_cnn), demo[test]]))[:, 1]
            log(f"nested fusion repeat {rep} fold {k}: {inner_splits} inner CNNs + one outer CNN")
    return tuple(result.mean(axis=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="CSV with PATNO, IMAGEID, y (0/1)")
    ap.add_argument("--fastsurfer-dir", required=True)
    ap.add_argument("--cache", required=True, help=".npy memmap of the pooled volumes (created if missing)")
    ap.add_argument("--out", required=True, help="CSV of out-of-fold probabilities")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pretrained", action="store_true", help="fine-tune the SFCN brain-age weights on 1 mm MNI volumes (batch 4, lr 1e-4)")
    a = ap.parse_args(argv)
    lab = pd.read_csv(a.labels, dtype={"IMAGEID": str}).dropna(subset=["y"])
    if a.pretrained:
        X = cache_volumes(a.fastsurfer_dir, lab["IMAGEID"].tolist(), a.cache, shape=MNI_SHAPE, loader=load_volume_mni)
        lab["p_cnn"] = cross_validate(X, lab["y"].to_numpy(), lab["PATNO"].to_numpy(), a.folds, a.repeats, a.epochs, a.device,
                                      model_fn=sfcn_pretrained, batch=4, lr=1e-4, patience=6)
    else:
        X = cache_volumes(a.fastsurfer_dir, lab["IMAGEID"].tolist(), a.cache)
        lab["p_cnn"] = cross_validate(X, lab["y"].to_numpy(), lab["PATNO"].to_numpy(), a.folds, a.repeats, a.epochs, a.device)
    lab.to_csv(a.out, index=False)
    from sklearn.metrics import roc_auc_score
    print(f"out-of-fold AUROC {roc_auc_score(lab['y'], lab['p_cnn']):.3f} (n={len(lab)})")


if __name__ == "__main__":
    main()
