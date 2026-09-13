import json
import os
from pathlib import Path

import pytest

from pie.imaging.staging import snapshot_tree


def source_tree(tmp_path):
    root=tmp_path/'source'
    root.mkdir()
    (root/'nested').mkdir()
    (root/'nested/a').write_bytes(b'original scientific bytes')
    os.link(root/'nested/a',root/'hardlink')
    (root/'relative').symlink_to('nested/a')
    (root/'absolute').symlink_to('/caller/chosen/reference')
    return root


def test_verified_bytes_links_timestamps_and_reuse(tmp_path):
    source=source_tree(tmp_path)
    out=tmp_path/'staged'
    saved=snapshot_tree(source,out,max_bytes=100)
    assert saved['status']=='verified' and saved['unique_source_bytes']==25
    assert (out/'nested/a').read_bytes()==(source/'nested/a').read_bytes()
    assert (out/'nested/a').stat().st_ino==(out/'hardlink').stat().st_ino
    assert (out/'nested/a').stat().st_mtime_ns==(source/'nested/a').stat().st_mtime_ns
    assert os.readlink(out/'absolute')=='/caller/chosen/reference'
    assert snapshot_tree(source,out,max_bytes=100)['entries']==saved['entries']
    (out/'nested/a').write_bytes(b'altered')
    with pytest.raises(ValueError,match='changed'):
        snapshot_tree(source,out,max_bytes=100)


def test_budget_unknown_destination_and_nesting_rejected(tmp_path):
    source=source_tree(tmp_path)
    with pytest.raises(ValueError,match='budget'):
        snapshot_tree(source,tmp_path/'budget',max_bytes=2)
    with pytest.raises(ValueError,match='nonnested'):
        snapshot_tree(source,source/'copy',max_bytes=100)
    out=tmp_path/'existing';out.mkdir();(out/'keep').write_text('keep')
    with pytest.raises(FileExistsError,match='unjournaled'):
        snapshot_tree(source,out,max_bytes=100)
    assert (out/'keep').read_text()=='keep'


def test_source_and_output_membership_changes_rejected(tmp_path):
    source=source_tree(tmp_path);out=tmp_path/'staged'
    snapshot_tree(source,out,max_bytes=100)
    (out/'unexpected').write_text('preserve me')
    with pytest.raises(ValueError,match='Unexpected'):
        snapshot_tree(source,out,max_bytes=100)
    assert (out/'unexpected').read_text()=='preserve me'


def test_guard_prevents_copy_and_partial_requires_review(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'a').write_bytes(b'abc')
    def stop():raise OSError('reserve')
    with pytest.raises(OSError,match='reserve'):
        snapshot_tree(source,tmp_path/'out',max_bytes=100,guard=stop)
    assert not (tmp_path/'out/a').exists()
    (tmp_path/'out/a.snapshot-partial').write_bytes(b'partial')
    with pytest.raises(FileExistsError):
        snapshot_tree(source,tmp_path/'out',max_bytes=100)
    assert (tmp_path/'out/a.snapshot-partial').read_bytes()==b'partial'


def test_completed_file_recovered_after_unflushed_journal(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'a').write_bytes(b'abc')
    out=tmp_path/'out'
    snapshot_tree(source,out,max_bytes=100)
    journal=tmp_path/'out.snapshot.json'
    saved=json.loads(journal.read_text());saved['entries']={};saved['status']='copying'
    journal.write_text(json.dumps(saved))
    assert snapshot_tree(source,out,max_bytes=100)['status']=='verified'


def test_explicit_include_does_not_copy_unrelated_files(tmp_path):
    source=source_tree(tmp_path)
    saved=snapshot_tree(source,tmp_path/'out',max_bytes=100,include=['nested','absolute'])
    assert set(saved['entries'])=={'nested','nested/a','absolute'}
    with pytest.raises(ValueError,match='immediate'):
        snapshot_tree(source,tmp_path/'other',max_bytes=100,include=['../outside'])


def test_links_recovered_after_unflushed_journal(tmp_path):
    source=source_tree(tmp_path);out=tmp_path/'out'
    snapshot_tree(source,out,max_bytes=100)
    journal=tmp_path/'out.snapshot.json'
    saved=json.loads(journal.read_text());saved['entries']={};saved['status']='copying'
    journal.write_text(json.dumps(saved))
    assert snapshot_tree(source,out,max_bytes=100)['status']=='verified'


def test_completed_snapshot_does_not_silently_expand(tmp_path):
    source=source_tree(tmp_path);out=tmp_path/'out'
    snapshot_tree(source,out,max_bytes=100)
    (source/'new').write_bytes(b'new acquisition')
    with pytest.raises(ValueError,match='membership changed'):
        snapshot_tree(source,out,max_bytes=100)
    assert not (out/'new').exists()


def test_requested_absent_source_is_not_a_complete_snapshot(tmp_path):
    source=source_tree(tmp_path)
    with pytest.raises(ValueError,match='child is missing'):
        snapshot_tree(source,tmp_path/'out',max_bytes=100,include=['missing'])


def test_changed_output_timestamp_is_detected(tmp_path):
    source=source_tree(tmp_path);out=tmp_path/'out'
    snapshot_tree(source,out,max_bytes=100)
    os.utime(out/'nested/a',ns=(1,1))
    with pytest.raises(ValueError,match='output metadata'):
        snapshot_tree(source,out,max_bytes=100)
