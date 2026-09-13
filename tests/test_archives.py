import os
import zipfile

import pytest

from pie.imaging.archives import ZipArchiveCache


def archive(tmp_path, name='one.zip'):
    path = tmp_path / name
    with zipfile.ZipFile(path, 'w') as stream:
        stream.writestr('z.dcm', b'first')
        stream.writestr('a.dcm', b'second')
    return path


def test_cache_reuses_index_preserves_order_and_closes(tmp_path):
    path = archive(tmp_path)
    with ZipArchiveCache() as cache:
        with cache.open(path) as first:
            assert [m.filename for m in first.infolist()] == ['z.dcm', 'a.dcm']
            assert [first.read(m) for m in first.infolist()] == [b'first', b'second']
        with cache.open(path) as second:
            assert second is first
        assert cache.stats()['hits'] == cache.stats()['misses'] == 1
    assert first.fp is None
    with pytest.raises(RuntimeError, match='closed'):
        with cache.open(path):
            pass


def test_lru_evicts_before_open_and_exception_closes(tmp_path):
    paths = [archive(tmp_path, f'{n}.zip') for n in range(3)]
    with pytest.raises(RuntimeError, match='test error'):
        with ZipArchiveCache(2) as cache:
            with cache.open(paths[0]) as first:
                pass
            with cache.open(paths[1]) as second:
                pass
            with cache.open(paths[0]):
                pass
            with cache.open(paths[2]) as third:
                assert second.fp is None and first.fp is not None
            assert cache.evictions == 1
            raise RuntimeError('test error')
    assert first.fp is third.fp is None


@pytest.mark.parametrize('during', [False, True])
def test_replaced_archive_with_same_size_and_mtime_fails(tmp_path, during):
    path = archive(tmp_path)
    def replace():
        before = path.stat()
        other = archive(tmp_path, 'replacement.zip')
        os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
        other.replace(path)
    with ZipArchiveCache() as cache:
        if during:
            with pytest.raises(ValueError, match='Archive changed'):
                with cache.open(path):
                    replace()
        else:
            with cache.open(path):
                pass
            replace()
            with pytest.raises(ValueError, match='Archive changed'):
                with cache.open(path):
                    pass


def test_corrupt_member_crc_is_still_checked(tmp_path):
    path = archive(tmp_path)
    data = path.read_bytes().replace(b'first', b'wrong')
    path.write_bytes(data)
    with ZipArchiveCache() as cache:
        with cache.open(path) as stream:
            with pytest.raises(zipfile.BadZipFile, match='CRC'):
                stream.read('z.dcm')


def test_nonoverlap_and_process_ownership(tmp_path, monkeypatch):
    path = archive(tmp_path)
    with ZipArchiveCache() as cache:
        with cache.open(path):
            with pytest.raises(RuntimeError, match='overlap'):
                with cache.open(path):
                    pass
            with pytest.raises(RuntimeError, match='during a lease'):
                cache.close()
        with monkeypatch.context() as patch:
            patch.setattr(os, 'getpid', lambda: -1)
            with pytest.raises(RuntimeError, match='processes or threads'):
                with cache.open(path):
                    pass


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        ZipArchiveCache(capacity)


def test_benchmark_fingerprints_match(tmp_path):
    from pie.imaging.archives import benchmark_zip_index
    result = benchmark_zip_index(archive(tmp_path), '', repeats=2)
    assert len(result['measurements']) == 4
    assert len({r['fingerprint'] for r in result['measurements']}) == 1
    assert all(r['members'] == 2 for r in result['measurements'])
