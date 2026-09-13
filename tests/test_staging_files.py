import hashlib
from pathlib import Path
import pytest
from pie.imaging.staging import snapshot_files


def setup(tmp_path):
    src = tmp_path / 'source'
    src.mkdir()
    (src / 'a').write_bytes(b'example')
    return src, tmp_path / 'target', {'a': hashlib.sha256(b'example').hexdigest()}


def test_copy_and_verified_reuse(tmp_path):
    src, dst, files = setup(tmp_path)
    result = snapshot_files(src, dst, files, max_bytes=7)
    assert result['total_bytes'] == 7 and (dst / 'a').read_bytes() == b'example'
    assert snapshot_files(src, dst, files, max_bytes=7) == result
    (dst / 'a').write_bytes(b'changed')
    with pytest.raises(ValueError, match='destination'):
        snapshot_files(src, dst, files, max_bytes=7)


def test_budget_and_path_escape(tmp_path):
    src, dst, files = setup(tmp_path)
    with pytest.raises(ValueError, match='budget'):
        snapshot_files(src, dst, files, max_bytes=6)
    with pytest.raises(ValueError, match='relative'):
        snapshot_files(src, dst, {'../escape': files['a']}, max_bytes=7)
    (src / 'outside').symlink_to(tmp_path / 'other')
    with pytest.raises(ValueError, match='symlink'):
        snapshot_files(src, dst, {'outside': files['a']}, max_bytes=7)


def test_mismatch_preserves_partial_and_original(tmp_path):
    src, dst, files = setup(tmp_path)
    with pytest.raises(ValueError, match='SHA256'):
        snapshot_files(src, dst, {'a': '0'*64}, max_bytes=7)
    assert (dst / 'a.partial').read_bytes() == b'example'
    assert not (dst / 'a').exists()
    with pytest.raises(FileExistsError):
        snapshot_files(src, dst, files, max_bytes=7)


def test_guard_prevents_writes(tmp_path):
    src, dst, files = setup(tmp_path)
    def guard():
        raise OSError('storage reserve')
    with pytest.raises(OSError, match='reserve'):
        snapshot_files(src, dst, files, max_bytes=7, guard=guard)
    assert not dst.exists()
