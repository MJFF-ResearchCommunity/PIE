"""Bounded immutable BOLD reads shared by outcome-independent QC variants."""
import os
from pathlib import Path
import threading

import nibabel as nib
import numpy as np


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class BOLDImageCache:
    """Hold at most one float32 4D image until context exit.

    No filesystem writes. Use one context per subject; returned arrays are
    read-only. A changed source fails closed instead of serving stale data.
    This trades one image's resident memory for avoiding repeated decompression.
    """

    def __init__(self):
        self._owner = os.getpid(), threading.get_ident()
        self._path = self._signature = self._data = None
        self._closed = False
        self.loads = 0

    def get(self, path):
        if self._closed or self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('Image cache is closed or accessed outside its owner')
        path = Path(path).resolve(strict=True)
        signature = _identity(path)
        if path == self._path:
            if signature != self._signature:
                raise ValueError('BOLD source changed while cached')
            return self._data
        self._data = self._path = self._signature = None
        image = nib.load(path)
        if image.ndim != 4:
            raise ValueError('BOLD cache requires a 4D image')
        data = image.get_fdata(dtype=np.float32)
        if _identity(path) != signature:
            raise ValueError('BOLD source changed during read')
        data.setflags(write=False)
        self._path, self._signature, self._data = path, signature, data
        self.loads += 1
        return data

    def close(self):
        self._data = self._path = self._signature = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Image cache is closed')
        return self

    def __exit__(self, *exc):
        self.close()
