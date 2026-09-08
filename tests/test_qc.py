"""Montage rendering on synthetic arrays (no data needed)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.qc import gallery, montage


def test_montage_and_gallery(tmp_path):
    base = np.zeros((30, 40, 20), dtype=np.float32)
    base[5:25, 8:32, 4:16] = 1.0
    mask = np.zeros_like(base, dtype=bool)
    mask[12:18, 18:24, 8:12] = True
    pngs = []
    for i in range(2):
        p = tmp_path / f"s{i}.png"
        montage(base, [(mask, "red")], p, title=f"s{i}", zoom=15)
        assert p.exists() and p.stat().st_size > 1000
        pngs.append(p)
    gallery(pngs, tmp_path / "gallery.png")
    assert (tmp_path / "gallery.png").exists()
