"""Verified, non-destructive directory staging with caller-selected storage.

Preserves file bytes, timestamps, symlinks and hardlinks for resumable scientific
work directories. Callers must quiesce writers before checkpoint snapshots.
No machine path, drive requirement, mount operation or implicit scratch location.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

from .fmri import sha256, write_json


def snapshot_files(source, destination, files, *, max_bytes, guard=None, progress=None):
    """Stage an explicit relative-path/SHA256 manifest without a directory scan.

    Sources are read-only. Each file is published only after its expected digest
    matches; existing destinations must also match. Interrupted partial files are
    preserved and refused, rather than silently overwritten. Callers select all
    paths, budgets and storage policy; no processing checkpoint is implied.
    """
    source, destination = Path(source), Path(destination)
    if not files or max_bytes <= 0:
        raise ValueError('Nonempty manifest and positive byte budget required')
    names = [Path(name) for name in files]
    if any(p.is_absolute() or '..' in p.parts or str(p) in ('.', '') for p in names):
        raise ValueError('Manifest paths must stay relative to their root')
    if any(len(d) != 64 or any(c not in '0123456789abcdef' for c in d) for d in files.values()):
        raise ValueError('Expected SHA256 digests required')
    srcroot, dstroot = source.resolve(), destination.resolve()
    if srcroot == dstroot or srcroot in dstroot.parents or dstroot in srcroot.parents:
        raise ValueError('Separate nonnested roots required')
    total, verified = 0, {}
    for name, expected in files.items():
        if guard:
            guard()
        src, dst = source / name, destination / name
        if not src.resolve().is_relative_to(srcroot) or not dst.resolve().is_relative_to(dstroot):
            raise ValueError('Manifest symlink escapes its root')
        if dst.is_symlink():
            raise ValueError('Destination symlink refused')
        before = src.stat()
        total += before.st_size
        if not stat.S_ISREG(before.st_mode) or total > max_bytes:
            raise ValueError('Nonregular source or manifest byte budget exceeded')
        if dst.exists():
            if not dst.is_file() or sha256(dst) != expected:
                raise ValueError('Existing destination does not match manifest')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            partial = dst.with_name(dst.name + '.partial')
            digest = hashlib.sha256()
            with src.open('rb') as incoming, partial.open('xb') as outgoing:
                copied = 0
                for block in iter(lambda: incoming.read(4 * 1024**2), b''):
                    if guard:
                        guard()
                    copied += len(block)
                    if copied > before.st_size:
                        raise ValueError('Source grew during staging')
                    outgoing.write(block)
                    digest.update(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = src.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError('Source changed during staging')
            if digest.hexdigest() != expected or sha256(partial) != expected:
                raise ValueError('Staged file SHA256 mismatch; partial preserved')
            os.link(partial, dst)  # Atomic, refusing overwrite even on a race.
            partial.unlink()
        verified[name] = expected
        if progress:
            progress(dict(path=name, verified_files=len(verified), total_bytes=total))
    return dict(files=verified, total_bytes=total, status='verified')


def snapshot_tree(source, destination, *, max_bytes, guard=None, progress=None, include=None):
    """Copy an immutable/quiesced tree, verifying copied bytes without deletion.

    A file-level journal permits reuse after interruption. Unjournaled destination
    trees and partial files are refused; completed files between journal writes
    are reused only after byte/metadata verification. Symlinks are
    copied as links (including absolute ones), never followed during traversal;
    their suitability inside a processing namespace remains the caller's duty.
    Budget counts logical bytes of distinct regular-file inodes. Source inode,
    size, mtime and ctime must remain stable during each read.
    """
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if max_bytes <= 0 or not source.is_dir():
        raise ValueError('Positive byte budget and directory source required')
    resolved = destination.resolve()
    if source == resolved or source in resolved.parents or resolved in source.parents:
        raise ValueError('Source/destination must be separate nonnested trees')
    if destination.is_symlink():
        raise ValueError('Symlink destination is not accepted')
    if include is not None and (not include or len(set(include)) != len(include) or any(
            not n or n in ('.','..') or len(Path(n).parts)!=1 or Path(n).is_absolute() for n in include)):
        raise ValueError('Include must name distinct immediate source children')
    journal = destination.parent/(destination.name+'.snapshot.json')
    saved = json.loads(journal.read_text()) if journal.exists() else None
    identity = dict(source=str(source), destination=str(destination), max_bytes=max_bytes,
                    include=sorted(include) if include is not None else None)
    if saved and saved['identity'] != identity:
        raise ValueError('Snapshot identity changed')
    if not saved and destination.exists():
        raise FileExistsError('Existing unjournaled destination requires review')
    if saved is None:
        saved = dict(identity=identity, entries={}, status='copying', source_writers_must_be_quiesced=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(journal,saved)
        destination.mkdir()
    elif not destination.is_dir():
        raise ValueError('Snapshot destination absent')
    frozen_members=set(saved['entries']) if saved['status']=='verified' else None
    budget, inodes, visited = 0, {}, set()
    started, last = time.monotonic(), time.monotonic()

    def check():
        if guard:
            guard()

    def tick(name, copied):
        nonlocal last
        if progress and time.monotonic()-last >= 10:
            progress(dict(path=name, copied_bytes=copied, unique_source_bytes=budget,
                          entries=len(visited), elapsed_seconds=time.monotonic()-started))
            last=time.monotonic()

    def walk(folder):
        # scandir uses directory-entry metadata and does not follow links.
        with os.scandir(folder) as scan:
            entries = sorted(scan,key=lambda e:e.name)
        if Path(folder)==source and include is not None and not set(include)<={e.name for e in entries}:
            raise ValueError('Explicitly included source child is missing')
        for entry in entries:
            if Path(folder)==source and include is not None and entry.name not in include:
                continue
            yield Path(entry.path),entry.stat(follow_symlinks=False)
            if entry.is_dir(follow_symlinks=False):
                yield from walk(entry.path)

    for path, before in walk(source):
        check()
        name = str(path.relative_to(source))
        if frozen_members is not None and name not in frozen_members:
            raise ValueError('Source tree membership changed')
        visited.add(name)
        out = destination/name
        prior = saved['entries'].get(name)
        if stat.S_ISDIR(before.st_mode):
            value = dict(kind='directory')
            if out.is_symlink() or (out.exists() and not out.is_dir()):
                raise ValueError('Destination directory type changed')
            out.mkdir(exist_ok=True)
        elif stat.S_ISLNK(before.st_mode):
            value = dict(kind='symlink', target=os.readlink(path))
            if prior:
                if prior != value or not out.is_symlink() or os.readlink(out) != value['target']:
                    raise ValueError('Snapshot symlink changed')
            elif out.is_symlink():
                if os.readlink(out) != value['target']:
                    raise ValueError('Unjournaled snapshot symlink changed')
            else:
                out.symlink_to(value['target'])
        elif stat.S_ISREG(before.st_mode):
            inode = (before.st_dev,before.st_ino)
            first = inodes.get(inode)
            if first is None:
                budget += before.st_size
                if budget > max_bytes:
                    raise ValueError('Snapshot exceeds caller byte budget')
                inodes[inode] = name
            signature = (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            if out.is_symlink():
                raise ValueError('Destination file replaced by symlink')
            if prior:
                digest = sha256(path)
                if prior.get('sha256') != digest or not out.is_file() or sha256(out) != digest:
                    raise ValueError('Snapshot source/output changed')
                if (out.stat().st_mtime_ns!=before.st_mtime_ns or
                    stat.S_IMODE(out.stat().st_mode)!=stat.S_IMODE(before.st_mode)):
                    raise ValueError('Snapshot output metadata changed')
            elif first is not None:
                digest = saved['entries'][first]['sha256']
                if out.exists():
                    if out.stat().st_ino != (destination/first).stat().st_ino or sha256(out) != digest:
                        raise ValueError('Unjournaled hardlink changed')
                else:
                    os.link(destination/first,out)
            elif out.exists():
                # A finished copy can precede the periodic journal checkpoint.
                # Reuse only after independently proving source/output identity.
                digest = sha256(path)
                if not out.is_file() or sha256(out) != digest:
                    raise ValueError('Unjournaled snapshot file differs from source')
                if out.stat().st_mtime_ns != before.st_mtime_ns:
                    raise ValueError('Unjournaled snapshot metadata differs from source')
            else:
                partial = out.with_name(out.name+'.snapshot-partial')
                hasher, count = hashlib.sha256(),0
                with path.open('rb') as src, partial.open('xb') as dest:
                    while block := src.read(4*1024**2):
                        check()
                        dest.write(block)
                        hasher.update(block)
                        count += len(block)
                        tick(name,count)
                    dest.flush()
                    os.fsync(dest.fileno())
                digest = hasher.hexdigest()
                if count != before.st_size or sha256(partial) != digest:
                    raise OSError('Snapshot byte verification failed')
                shutil.copystat(path,partial,follow_symlinks=False)
                partial.rename(out)
            after = path.stat()
            if signature != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('Source changed during snapshot')
            value = dict(kind='file',bytes=before.st_size,sha256=digest,
                         mtime_ns=before.st_mtime_ns,mode=stat.S_IMODE(before.st_mode),hardlink_to=first)
            if prior and prior != value:
                raise ValueError('Source metadata/hardlink identity changed')
        else:
            raise ValueError('Unsupported special file in scientific snapshot')
        saved['entries'][name]=value
        # Avoid quadratic full-manifest rewrites for thousands of small files.
        # Unjournaled completed files are adopted only by verified reuse above.
        if len(visited) % 64 == 0:
            write_json(journal,saved)
        tick(name,0)
    if visited != set(saved['entries']):
        raise ValueError('Source tree membership changed')
    actual = {str(p.relative_to(destination)) for p,_ in walk(destination)}
    if actual != visited:
        raise ValueError('Unexpected snapshot destination files')
    # Restore directory timestamps after writing their children.
    for name in sorted(visited,key=lambda n:len(Path(n).parts),reverse=True):
        if saved['entries'][name]['kind']=='directory':
            shutil.copystat(source/name,destination/name,follow_symlinks=False)
    saved.update(status='verified', unique_source_bytes=budget, elapsed_seconds=time.monotonic()-started)
    write_json(journal,saved)
    return saved
