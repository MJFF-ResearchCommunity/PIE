"""Conservative temporal assembly of converter-split classic DICOM volumes.

Requires explicit temporal identifiers, complete slice geometry and unique
acquisition-time matches. Never orders by filenames or invents acquisition
metadata. All input/output paths and the memory budget are caller supplied.
"""
from datetime import datetime
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np

from .fmri import inspect_nifti, sha256, write_json

_CODE_SHA256 = sha256(__file__)


def time_key(value):
    value = str(value).replace(':', '')
    if not re.fullmatch(r'\d{6}(?:\.\d{1,6})?', value):
        raise ValueError('Explicit acquisition clock time is required')
    parsed = datetime.strptime(value, '%H%M%S.%f' if '.' in value else '%H%M%S')
    return ((parsed.hour*60+parsed.minute)*60+parsed.second)*1_000_000+parsed.microsecond


def temporal_geometry(audit):
    """Return ordered evidence, enforcing a complete classic slice stack per TPI."""
    if not audit.get('selected_members_crc_checked') or audit.get('duplicate_sop_uids'):
        raise ValueError('CRC-checked unique DICOM evidence is required')
    records = audit['records']
    if len(records) != audit['dicom_count'] or not records:
        raise ValueError('Incomplete header evidence')
    groups, uids = {}, set()
    constants = ['StudyInstanceUID','SeriesInstanceUID','FrameOfReferenceUID',
                 'ImageOrientationPatient','PixelSpacing','Rows','Columns','SliceThickness',
                 'SpacingBetweenSlices','RepetitionTime','EchoTime']
    signatures = set()
    for row in records:
        h = row['header']
        t = h.get('TemporalPositionIdentifier')
        if not isinstance(t, int) or isinstance(t, bool) or t < 1:
            raise ValueError('Explicit positive integer temporal position is required')
        if h.get('NumberOfFrames') not in (None, 1):
            raise ValueError('Enhanced/multiframe inputs require another validated mapping')
        uid = h.get('SOPInstanceUID')
        if not uid or uid in uids:
            raise ValueError('Missing or duplicate SOP identity')
        uids.add(uid)
        if any(h.get(k) is None for k in constants):
            raise ValueError('Incomplete geometry/timing evidence')
        signatures.add(json.dumps([h[k] for k in constants], sort_keys=True))
        groups.setdefault(t, []).append(h)
    if len(signatures) != 1 or sorted(groups) != list(range(1, len(groups)+1)) or len(groups) < 2:
        raise ValueError('Inconsistent geometry/timing or missing temporal positions')
    geometry, ordered = None, []
    for t, headers in sorted(groups.items()):
        if any(h.get('NumberOfTemporalPositions') != len(groups) for h in headers):
            raise ValueError('Declared temporal count disagrees with complete audit')
        positions = [tuple(h['ImagePositionPatient']) for h in headers]
        if len(set(positions)) != len(positions) or len(positions) < 2:
            raise ValueError('Duplicate/missing spatial positions in a volume')
        if any(len(p) != 3 or not np.isfinite(p).all() for p in positions):
            raise ValueError('Invalid slice position')
        signature = set(positions)
        if geometry is None:
            geometry = signature
            orientation = np.asarray(headers[0]['ImageOrientationPatient'], float)
            if orientation.shape != (6,):
                raise ValueError('Invalid DICOM orientation')
            normal = np.cross(orientation[:3], orientation[3:])
            if not np.isclose(np.linalg.norm(normal), 1, atol=1e-5):
                raise ValueError('Invalid DICOM orientation vectors')
            locations = np.sort(np.asarray(positions) @ normal)
            spacing = float(headers[0]['SpacingBetweenSlices'])
            if spacing <= 0 or not np.allclose(np.diff(locations), spacing, rtol=0, atol=1e-4):
                raise ValueError('Missing or irregular slice geometry')
        elif geometry != signature:
            raise ValueError('Temporal volumes have different spatial coverage')
        times = {(str(h.get('AcquisitionDate')), time_key(h.get('AcquisitionTime'))) for h in headers}
        if len(times) != 1:
            raise ValueError('Slice-specific timing cannot be matched to one converted volume')
        date, clock = next(iter(times))
        day = datetime.strptime(date, '%Y%m%d').toordinal()
        triggers = {h.get('TriggerTime') for h in headers}
        ordered.append(dict(temporal_position=t, clock_microseconds=clock,
                            absolute_microseconds=day*86400_000000+clock,
                            dicom_trigger_time_ms=next(iter(triggers)) if len(triggers)==1 else None))
    if len({x['clock_microseconds'] for x in ordered}) != len(ordered):
        raise ValueError('Acquisition clock time is not a unique volume identifier')
    return ordered, records[0]['header'], len(geometry)


def assemble_temporal_volumes(conversion_dir, header_audit, output_dir, *,
                              max_memory_bytes=1024*1024**2, timing_tolerance_seconds=.001,
                              allow_verified_trigger_time_partition=False):
    """Preserve physical voxel values in a new 4D NIfTI and provenance record.

    Inputs remain unchanged. A nonempty or incomplete destination is never
    overwritten. No scratch fallback, mount requirement, or default data path.
    Temporal assembly alone is not preprocessing or scientific inclusion.
    An explicit trigger-partition option permits removal of a volume-specific
    converter TriggerDelayTime only when its raw value matches that volume's
    DICOM TriggerTime (millisecond tag, within integer-rounding precision).
    Every raw value is preserved in provenance; it is not exported as a
    seconds-valued run-level timing parameter. AcquisitionTime establishes TR.
    """
    source = Path(conversion_dir).resolve(strict=True)
    evidence_path = Path(header_audit).resolve(strict=True)
    destination = Path(output_dir).absolute()
    if max_memory_bytes <= 0 or not 0 <= timing_tolerance_seconds <= .001:
        raise ValueError('Invalid memory budget or overly permissive timing tolerance')
    conversion_path = source/'conversion.json'
    converted = json.loads(conversion_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    identity = dict(conversion_json_sha256=sha256(conversion_path), audit_sha256=sha256(evidence_path),
                    timing_tolerance_seconds=timing_tolerance_seconds, code_sha256=_CODE_SHA256,
                    allow_verified_trigger_time_partition=allow_verified_trigger_time_partition)
    if (evidence['source_member_index_sha256'] != converted['source_member_index_sha256'] or
        any(converted['source'].get(k) != v for k,v in evidence['source'].items()) or
        not converted.get('selected_members_crc_checked')):
        raise ValueError('Conversion and header audit refer to different source members')
    ordered, header, slices = temporal_geometry(evidence)
    if (header['StudyInstanceUID'] != converted['study_uid'] or
            header['SeriesInstanceUID'] != converted['series_uid']):
        raise ValueError('DICOM/converted acquisition identity mismatch')
    if len(converted['outputs']) != len(ordered):
        raise ValueError('Converted output count differs from temporal-position count')
    tr = float(header['RepetitionTime'])/1000
    intervals = np.diff([r['absolute_microseconds'] for r in ordered])/1e6
    if not np.isfinite(tr) or tr <= 0 or not np.allclose(intervals, tr, atol=timing_tolerance_seconds, rtol=0):
        raise ValueError('Acquisition timestamps disagree with declared repetition time')
    by_clock, common, first = {}, None, None
    ignored = {'SeriesNumber', 'AcquisitionTime', 'BidsGuess'}
    if allow_verified_trigger_time_partition:
        ignored.add('TriggerDelayTime')
    evidence_by_clock = {r['clock_microseconds']:r for r in ordered}
    for row in converted['outputs']:
        for k in ['nifti','sidecar']:
            path = (source/row[k]).resolve(strict=True)
            if not path.is_relative_to(source) or sha256(path) != row[k+'_sha256']:
                raise ValueError('Converted input path or hash changed')
        metadata = json.loads((source/row['sidecar']).read_text())
        clock = time_key(metadata.get('AcquisitionTime'))
        if clock in by_clock:
            raise ValueError('Duplicate converted acquisition time')
        if allow_verified_trigger_time_partition:
            raw_trigger = evidence_by_clock.get(clock,{}).get('dicom_trigger_time_ms')
            exported_trigger = metadata.get('TriggerDelayTime')
            if raw_trigger is None or not np.isfinite(raw_trigger):
                raise ValueError('Trigger partition lacks unique DICOM evidence')
            if exported_trigger is None:
                if raw_trigger != 0:
                    raise ValueError('Nonzero DICOM trigger missing from converter metadata')
            elif not np.isclose(float(exported_trigger),float(raw_trigger),atol=.51,rtol=0):
                raise ValueError('Converter trigger field does not match raw DICOM millisecond tag')
            evidence_by_clock[clock]['converter_trigger_delay_time_raw'] = exported_trigger
        stable = {k:v for k,v in metadata.items() if k not in ignored}
        if common is None:
            common = stable
        elif stable != common:
            changed = sorted(k for k in set(stable) | set(common) if stable.get(k) != common.get(k))
            raise ValueError('Unresolved metadata differences between converted volumes: '+', '.join(changed))
        if not np.isclose(float(metadata.get('RepetitionTime', np.nan)), tr, atol=1e-6, rtol=0):
            raise ValueError('Sidecar repetition time disagrees with DICOM')
        if not np.isclose(float(metadata.get('EchoTime', np.nan)), float(header['EchoTime'])/1000, atol=1e-6, rtol=0):
            raise ValueError('Sidecar echo time disagrees with DICOM')
        img = nib.load(source/row['nifti'])
        if img.ndim != 3 or img.header.get_xyzt_units()[0] != 'mm' or not np.isfinite(img.affine).all():
            raise ValueError('Expected a spatially valid 3D millimeter NIfTI')
        if sorted(img.shape) != sorted([header['Rows'],header['Columns'],slices]):
            raise ValueError('Converted spatial dimensions disagree with slice audit')
        if img.get_data_dtype().kind not in 'iuf' or (img.get_data_dtype().kind in 'iu' and img.get_data_dtype().itemsize > 4):
            raise ValueError('Unsupported voxel type for lossless physical-value assembly')
        if first is None:
            first = img
        elif (img.shape != first.shape or not np.array_equal(img.affine, first.affine) or
              not np.array_equal(img.header.get_zooms(), first.header.get_zooms())):
            raise ValueError('Converted volumes have different geometry')
        by_clock[clock] = row
    if set(by_clock) != {r['clock_microseconds'] for r in ordered}:
        raise ValueError('DICOM/converted time mapping is not one-to-one')
    for entry in ordered:
        entry.update(by_clock[entry['clock_microseconds']])
    if destination.exists():
        marker = destination/'assembly.json'
        if not marker.is_file():
            raise FileExistsError('Incomplete/existing assembly destination must be reviewed')
        saved = json.loads(marker.read_text())
        if saved['identity'] != identity:
            raise ValueError('Existing assembly identity differs')
        for name,digest in saved['output_hashes'].items():
            if sha256(destination/name) != digest:
                raise ValueError('Existing assembly output changed')
        return json.loads((destination/'conversion.json').read_text())
    voxels = int(np.prod(first.shape))
    if (2*len(ordered)+2)*voxels*8 > max_memory_bytes:
        raise ValueError('Assembly exceeds configured memory budget')
    data = np.empty((*first.shape,len(ordered)),dtype=np.float64)
    for k,entry in enumerate(ordered):
        values = np.asanyarray(nib.load(source/entry['nifti']).dataobj)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite voxel data')
        data[...,k] = values
        if not np.array_equal(data[...,k], values):
            raise ValueError('Voxel values cannot be represented without loss')
    destination.mkdir(parents=True, exist_ok=False)
    image = nib.Nifti1Image(data, first.affine, first.header.copy())
    image.set_data_dtype(np.float64)
    image.header.set_slope_inter(1,0)
    image.header.set_xyzt_units('mm','sec')
    image.header.set_zooms((*first.header.get_zooms(),tr))
    nii,sidecar = destination/'assembled.nii.gz', destination/'assembled.json'
    nib.save(image,nii)
    common['AcquisitionTime'] = json.loads((source/ordered[0]['sidecar']).read_text())['AcquisitionTime']
    write_json(sidecar,common)
    check = nib.load(nii)
    if not np.array_equal(check.affine, first.affine):
        raise ValueError('Serialized assembly geometry changed')
    # One sequential decompression, not N repeated gzip scans from the beginning.
    serialized = np.asanyarray(check.dataobj)
    for k in range(len(ordered)):
        if not np.array_equal(serialized[...,k], data[...,k]):
            raise ValueError('Serialized assembly changed physical voxel values')
    output = dict(nifti=nii.name,sidecar=sidecar.name,nifti_sha256=sha256(nii),
                  sidecar_sha256=sha256(sidecar),**inspect_nifti(nii,sidecar))
    record = dict(converted, outputs=[output], temporal_assembly_identity=identity,
                  original_conversion_dir=str(source), header_audit=str(evidence_path))
    write_json(destination/'conversion.json',record)
    write_json(destination/'assembly.json',dict(identity=identity, ordered_inputs=ordered,
        physical_voxel_values_exactly_preserved=True, scientific_qc_pass=False,
        omitted_volume_specific_metadata=sorted(ignored),
        source_metadata_not_invented=True, output_hashes={p.name:sha256(p) for p in [nii,sidecar,destination/'conversion.json']}))
    return record
