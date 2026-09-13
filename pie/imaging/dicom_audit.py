"""Read-only, bounded classic-DICOM evidence for conversion failure review.

This audits archived headers and member integrity. It neither repairs metadata,
orders volumes by filename/InstanceNumber, nor declares scientific eligibility.
"""
from collections import Counter
from collections.abc import Sequence
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import warnings

import pydicom

from .archives import ZipArchiveCache

_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

TAGS = (
    'SOPClassUID', 'SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID',
    'FrameOfReferenceUID', 'InstanceNumber', 'TemporalPositionIdentifier',
    'NumberOfTemporalPositions', 'AcquisitionNumber', 'AcquisitionDate',
    'AcquisitionTime', 'AcquisitionDateTime', 'ContentTime', 'TriggerTime',
    'RepetitionTime', 'EchoTime', 'EchoNumbers', 'FlipAngle', 'NumberOfFrames',
    'Rows', 'Columns', 'ImagePositionPatient', 'ImageOrientationPatient',
    'PixelSpacing', 'SpacingBetweenSlices', 'SliceThickness', 'SliceLocation',
    'ImageType', 'ProtocolName', 'SeriesDescription', 'SequenceName',
    'MRAcquisitionType', 'ScanningSequence', 'SequenceVariant', 'ScanOptions',
    'InPlanePhaseEncodingDirection', 'PatientPosition', 'RescaleSlope',
    'RescaleIntercept', 'BitsAllocated', 'PixelRepresentation', 'SamplesPerPixel',
    'Manufacturer', 'ManufacturerModelName', 'ReceiveCoilName',
)


def _value(value):
    if value is None or isinstance(value, (str, int, float)):
        return value
    if isinstance(value, Sequence):
        return [_value(v) for v in value]
    return str(value)


def audit_archive_headers(archive, prefix, *, expected_count=None, archive_cache=None,
                          max_members=20000, max_member_bytes=64*1024**2, progress=None):
    """CRC-read selected members one at a time, without extracting or decoding pixels.

    The complete selected-member bytes are read to verify ZIP CRC and SHA256;
    only whitelisted header fields survive in the returned evidence. Size bounds
    deliberately reject giant enhanced objects; this is a classic-DICOM audit.
    ``progress(completed, total)`` is an optional caller-owned reporting hook.
    """
    path = Path(archive).resolve(strict=True)
    if (not prefix or not prefix.endswith('/') or PurePosixPath(prefix).is_absolute()
            or '..' in PurePosixPath(prefix).parts):
        raise ValueError('Unsafe archive prefix')
    if max_members < 1 or max_member_bytes < 1:
        raise ValueError('Audit size limits must be positive')
    started = time.monotonic()
    before = path.stat()
    rows, counts = [], {tag: Counter() for tag in TAGS}
    digest = hashlib.sha256()
    byte_count = 0
    warning_counts, warning_examples = Counter(), []
    with (ZipArchiveCache() if archive_cache is None else nullcontext(archive_cache)) as cache, cache.open(path) as z:
        members = [m for m in z.infolist() if m.filename.startswith(prefix)
                   and m.filename.lower().endswith('.dcm') and not m.is_dir()]
        if not members or len(members) > max_members:
            raise ValueError('Empty or oversized selected series')
        if len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate archive member paths')
        if expected_count is not None and len(members) != expected_count:
            raise ValueError('Selected count differs from inventory')
        if any(not 0 < m.file_size <= max_member_bytes for m in members):
            raise ValueError('Empty or oversized DICOM member')
        for i, member in enumerate(members):
            payload = z.read(member)  # Full selected-member CRC; no disk scratch.
            # Anonymized UID validation can emit thousands of distinct warnings.
            # Keep bounded evidence instead of flooding caller logs; do not alter
            # validation settings or normalize the actual opaque header values.
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter('always')
                dataset = pydicom.dcmread(io.BytesIO(payload), stop_before_pixels=True, specific_tags=TAGS)
                header = {tag: _value(dataset.get(tag)) for tag in TAGS}
            for warning in emitted:
                warning_counts[warning.category.__name__] += 1
                if len(warning_examples) < 5:
                    warning_examples.append(dict(category=warning.category.__name__,
                                                 message=str(warning.message)[:500], member=member.filename))
            for tag, value in header.items():
                counts[tag][json.dumps(value, sort_keys=True, allow_nan=False)] += 1
            rows.append(dict(member=member.filename, crc32=member.CRC, bytes=len(payload),
                             sha256=hashlib.sha256(payload).hexdigest(), header=header))
            byte_count += len(payload)
            digest.update(json.dumps([member.filename, member.CRC, member.file_size]).encode())
            del payload, dataset
            if progress is not None and ((i+1) % 250 == 0 or i+1 == len(members)):
                progress(i+1, len(members))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ino, after.st_ctime_ns):
        raise ValueError('Archive changed during header audit')
    summaries = {tag: dict(unique_values=len(counter), missing=counter.get('null', 0),
                          examples=[dict(value=json.loads(value), count=count)
                                    for value, count in counter.most_common(8)])
                 for tag, counter in counts.items()}
    sop = counts['SOPInstanceUID']
    return dict(source=dict(archive=str(path), size=before.st_size, mtime_ns=before.st_mtime_ns,
                            series_prefix=prefix), dicom_count=len(rows), dicom_bytes=byte_count,
                selected_members_crc_checked=True, source_member_index_sha256=digest.hexdigest(),
                duplicate_sop_uids={json.loads(k): v for k, v in sop.items() if k != 'null' and v > 1},
                tags=summaries, records=rows, elapsed_seconds=time.monotonic()-started,
                code_sha256=_CODE_SHA256,
                parser_warnings=dict(counts=dict(warning_counts), examples=warning_examples,
                                     examples_limit=5, metadata_repaired=False),
                scientific_qc_pass=False, metadata_modified=False, temporal_order_assumed=False)
