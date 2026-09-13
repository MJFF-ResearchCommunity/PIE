"""Bounded, explicitly owned read-only ZIP indexes for sequential conversion.

The cache never caches extracted bytes or disables ZipFile's member CRC checks.
One million-entry index can exceed a gigabyte: callers choose a small capacity.
"""
from collections import OrderedDict
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time
import zipfile


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class ZipArchiveCache:
    """LRU cache of open indexes, confined to its creating process and thread.

    Use ``with ZipArchiveCache(max_open=1) as cache:`` and borrow handles with
    ``with cache.open(path) as archive:``. Leases may not overlap; handles must
    not escape their lease. Changes to a resident archive fail closed, including
    inode replacement with unchanged size and modification time.
    """

    def __init__(self, max_open=1):
        if isinstance(max_open, bool) or not isinstance(max_open, int) or max_open < 1:
            raise ValueError("max_open must be a positive integer")
        self.max_open = max_open
        self._owner = (os.getpid(), threading.get_ident())
        self._entries = OrderedDict()
        self._active = False
        self._closed = False
        self.hits = self.misses = self.evictions = 0
        self.index_seconds = 0.0

    def _check_owner(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("ZIP cache cannot be shared across processes or threads")
        if self._closed:
            raise RuntimeError("ZIP cache is closed")

    @staticmethod
    def _validate(path, archive, identity):
        if (archive.fp is None or _identity(path.stat()) != identity or
                _identity(os.fstat(archive.fp.fileno())) != identity):
            raise ValueError("Archive changed while cached or in use")

    @contextmanager
    def open(self, path):
        self._check_owner()
        if self._active:
            raise RuntimeError("ZIP cache leases must not overlap")
        path = Path(path).resolve(strict=True)
        if path in self._entries:
            archive, identity = self._entries[path]
            try:
                self._validate(path, archive, identity)
            except Exception:
                self._entries.pop(path)[0].close()
                raise
            self._entries.move_to_end(path)
            self.hits += 1
        else:
            # Evict before opening, bounding peak index memory as well as handles.
            if len(self._entries) >= self.max_open:
                self._entries.popitem(last=False)[1][0].close()
                self.evictions += 1
            identity = _identity(path.stat())
            started = time.perf_counter()
            archive = zipfile.ZipFile(path)
            try:
                self._validate(path, archive, identity)
            except Exception:
                archive.close()
                raise
            self.index_seconds += time.perf_counter() - started
            self.misses += 1
            self._entries[path] = (archive, identity)
        self._active = True
        try:
            yield archive
        finally:
            self._active = False
            try:
                self._validate(path, archive, identity)
            except Exception:
                self._entries.pop(path)[0].close()
                raise

    def close(self):
        if self._closed:
            return
        self._check_owner()
        if self._active:
            raise RuntimeError("Cannot close ZIP cache during a lease")
        for archive, _ in self._entries.values():
            archive.close()
        self._entries.clear()
        self._closed = True

    def __enter__(self):
        self._check_owner()
        return self

    def __exit__(self, *exc):
        self.close()

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    index_seconds=self.index_seconds, max_open=self.max_open)


def benchmark_zip_index(path, prefix, repeats=3):
    """Read-only cold-index versus retained-index benchmark, not extraction time.

    OS disk caches are not flushed. Both modes scan the same members in archive
    order; fingerprints must agree. Cold here means a new Python ZIP index.
    """
    import hashlib
    import json
    if repeats < 2:
        raise ValueError('At least two repetitions are required')
    rows = []
    for mode in ('reopen', 'cached'):
        with ZipArchiveCache() as shared:
            for repeat in range(repeats):
                started = time.perf_counter()
                with (ZipArchiveCache() if mode == 'reopen' else _borrow(shared)) as cache:
                    with cache.open(path) as stream:
                        opened = time.perf_counter()
                        members = [m for m in stream.infolist() if m.filename.startswith(prefix)
                                   and m.filename.lower().endswith('.dcm') and not m.is_dir()]
                        if not members:
                            raise ValueError('No selected DICOM members')
                        digest = hashlib.sha256()
                        for member in members:
                            digest.update(json.dumps([member.filename, member.CRC, member.file_size]).encode())
                        scanned = time.perf_counter()
                rows.append(dict(mode=mode, repeat=repeat, open_seconds=opened-started,
                                 selection_seconds=scanned-opened, total_seconds=scanned-started,
                                 members=len(members), fingerprint=digest.hexdigest()))
    if len({row['fingerprint'] for row in rows}) != 1:
        raise ValueError('Benchmark selected different member indexes')
    return dict(scope='ZIP index and selection only; OS cache not flushed; no output conversion',
                path=str(Path(path).resolve()), prefix=prefix, measurements=rows)


@contextmanager
def _borrow(cache):
    yield cache


def benchmark_zip_workload(requests, capacities=(1, 2)):
    """Replay explicit (archive, prefix) requests without extracting/writing data.

    Requests should represent the caller's real conversion order. Measures index
    residency tradeoffs, not conversion or complete preprocessing throughput.
    """
    import hashlib
    import json
    requests = list(requests)
    if not requests:
        raise ValueError('Need archive/prefix requests')
    results = []
    for capacity in capacities:
        started = time.perf_counter()
        fingerprints = []
        with ZipArchiveCache(capacity) as cache:
            for path, prefix in requests:
                with cache.open(path) as archive:
                    digest = hashlib.sha256()
                    count = 0
                    for member in archive.infolist():
                        if (member.filename.startswith(prefix) and
                                member.filename.lower().endswith('.dcm') and not member.is_dir()):
                            digest.update(json.dumps([member.filename, member.CRC, member.file_size]).encode())
                            count += 1
                    if not count:
                        raise ValueError('No DICOM members for workload request')
                    fingerprints.append(dict(count=count, sha256=digest.hexdigest()))
                # Do not retain an evicted ZipFile through a benchmark local.
                del archive
            results.append(dict(capacity=capacity, seconds=time.perf_counter()-started,
                                cache=cache.stats(), fingerprints=fingerprints))
    if any(row['fingerprints'] != results[0]['fingerprints'] for row in results):
        raise ValueError('Archive workload fingerprints changed')
    return dict(requests=[list(r) for r in requests], measurements=results,
                scope='Index/selection replay only; OS caches not flushed')
