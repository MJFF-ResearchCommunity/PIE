import io
import zipfile

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
import pytest

from pie.imaging.dicom_audit import audit_archive_headers


def archive_fixture(tmp_path, duplicate=False):
    path = tmp_path / 'series.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for i, instance in enumerate([8, 2, 5, 1]):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = f'1.2.3.{1 if duplicate else i+1}'
            ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = '1.2.3.10'
            ds.SeriesInstanceUID = '1.2.3.11'
            ds.InstanceNumber = instance
            ds.TemporalPositionIdentifier = 1 + i // 2
            ds.ImagePositionPatient = [0, 0, i % 2 * 3.5]
            ds.RepetitionTime = 2500
            ds.ProtocolName = 'example'
            ds.PatientName = 'Must not be included'
            ds.Rows = ds.Columns = 2
            ds.BitsAllocated = 16
            ds.PixelData = b'\0'*8
            stream = io.BytesIO()
            ds.save_as(stream, enforce_file_format=True)
            z.writestr(f'series/{i}.dcm', stream.getvalue())
    return path


def test_audit_preserves_metadata_not_assumed_order(tmp_path):
    path = archive_fixture(tmp_path)
    before = path.read_bytes()
    progress = []
    result = audit_archive_headers(path, 'series/', expected_count=4,
                                   progress=lambda a,b: progress.append((a,b)))
    assert result['selected_members_crc_checked'] and not result['metadata_modified']
    assert not result['scientific_qc_pass'] and not result['temporal_order_assumed']
    assert [r['header']['InstanceNumber'] for r in result['records']] == [8,2,5,1]
    assert result['tags']['TemporalPositionIdentifier']['unique_values'] == 2
    assert result['tags']['AcquisitionTime']['missing'] == 4
    assert all('PatientName' not in r['header'] and 'PixelData' not in r['header'] for r in result['records'])
    assert path.read_bytes() == before and progress == [(4,4)]


def test_bounds_inventory_and_duplicate_uid_evidence(tmp_path):
    path = archive_fixture(tmp_path, duplicate=True)
    assert audit_archive_headers(path, 'series/')['duplicate_sop_uids'] == {'1.2.3.1':4}
    for kwargs in [dict(expected_count=3), dict(max_members=3), dict(max_member_bytes=1)]:
        with pytest.raises(ValueError):
            audit_archive_headers(path, 'series/', **kwargs)
    with pytest.raises(ValueError, match='Unsafe'):
        audit_archive_headers(path, '../series/')


def test_parser_warnings_are_bounded_evidence_not_log_flood(tmp_path, monkeypatch):
    import warnings
    from pie.imaging import dicom_audit
    path = archive_fixture(tmp_path)
    original = dicom_audit.pydicom.dcmread

    def noisy_read(*args, **kwargs):
        for i in range(10):
            warnings.warn(f'Synthetic validation warning {i}', UserWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(dicom_audit.pydicom, 'dcmread', noisy_read)
    with warnings.catch_warnings(record=True) as external:
        result = audit_archive_headers(path, 'series/')
    assert not external
    evidence = result['parser_warnings']
    assert evidence['counts'] == {'UserWarning': 40}
    assert len(evidence['examples']) == 5 and not evidence['metadata_repaired']
    assert result['code_sha256'] == dicom_audit._CODE_SHA256
