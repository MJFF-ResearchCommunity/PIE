import json
import nibabel as nib
import numpy as np
import pytest

from pie.imaging.fmri import sha256
from pie.imaging.fmri_assembly import assemble_temporal_volumes, temporal_geometry, time_key


def fixture(tmp_path, n=3):
    source = tmp_path/'converted volumes'
    source.mkdir()
    records, outputs = [], []
    identity = {'archive':'caller-source.zip','size':123,'mtime_ns':456,'series_prefix':'caller-prefix/'}
    for t in range(n):
        # Names intentionally oppose chronological order.
        name = f'volume-{n-t}'
        img = nib.Nifti1Image(np.full((2,2,2),t+1,dtype=np.int16),np.diag([2.,2.,2.,1.]))
        img.header.set_xyzt_units('mm')
        nii, js = source/(name+'.nii.gz'), source/(name+'.json')
        nib.save(img,nii)
        js.write_text(json.dumps({'AcquisitionTime':f'12:00:{t*2:02}.000000','RepetitionTime':2.,
            'EchoTime':.03,'SeriesNumber':n-t,'Manufacturer':'example','PhaseEncodingAxis':'i'}))
        outputs.append(dict(nifti=nii.name,sidecar=js.name,nifti_sha256=sha256(nii),sidecar_sha256=sha256(js)))
        for z in range(2):
            h=dict(SOPInstanceUID=f'sop-{t}-{z}',TemporalPositionIdentifier=t+1,NumberOfTemporalPositions=n,
                NumberOfFrames=None,StudyInstanceUID='study',SeriesInstanceUID='series',FrameOfReferenceUID='frame',
                ImageOrientationPatient=[1,0,0,0,1,0],ImagePositionPatient=[0,0,z*2],PixelSpacing=[2,2],
                Rows=2,Columns=2,SliceThickness=2,SpacingBetweenSlices=2,RepetitionTime=2000,EchoTime=30,
                AcquisitionDate='20260101',AcquisitionTime=f'1200{t*2:02}')
            records.append({'header':h})
    conversion = dict(source=identity,source_member_index_sha256='members',selected_members_crc_checked=True,
        dicom_count=len(records),study_uid='study',series_uid='series',outputs=list(reversed(outputs)))
    (source/'conversion.json').write_text(json.dumps(conversion))
    audit = dict(source=identity,source_member_index_sha256='members',selected_members_crc_checked=True,
        duplicate_sop_uids={},dicom_count=len(records),records=list(reversed(records)))
    path = tmp_path/'audit.json'
    path.write_text(json.dumps(audit))
    return source,path,audit


def test_explicit_order_exact_pixels_and_no_invented_polarity(tmp_path):
    source,path,_ = fixture(tmp_path)
    result = assemble_temporal_volumes(source,path,tmp_path/'result')
    image = nib.load(tmp_path/'result/assembled.nii.gz')
    assert image.shape==(2,2,2,3)
    assert image.header.get_zooms()[3]==2
    np.testing.assert_array_equal(image.get_fdata()[0,0,0],[1,2,3])
    metadata = json.loads((tmp_path/'result/assembled.json').read_text())
    assert metadata['PhaseEncodingAxis']=='i'
    assert 'PhaseEncodingDirection' not in metadata and 'SliceTiming' not in metadata
    assert result['outputs'][0]['run_class']=='short_reference_candidate'
    assert assemble_temporal_volumes(source,path,tmp_path/'result')==result


@pytest.mark.parametrize('defect',['missing_tpi','missing_slice','wrong_tr','duplicate_sop','same_time','inconsistent_geometry'])
def test_incomplete_or_inconsistent_evidence_rejected(tmp_path,defect):
    source,path,audit = fixture(tmp_path)
    if defect=='missing_tpi': audit['records'][0]['header']['TemporalPositionIdentifier']=None
    if defect=='missing_slice': audit['records'].pop();audit['dicom_count']-=1
    if defect=='wrong_tr':
        for row in audit['records']:row['header']['RepetitionTime']=2100
    if defect=='duplicate_sop':audit['records'][1]['header']['SOPInstanceUID']=audit['records'][0]['header']['SOPInstanceUID']
    if defect=='same_time':
        for row in audit['records']:row['header']['AcquisitionTime']='120000'
    if defect=='inconsistent_geometry':audit['records'][0]['header']['PixelSpacing']=[3,3]
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):assemble_temporal_volumes(source,path,tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_changed_source_and_memory_limit_rejected(tmp_path):
    source,path,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='memory'):assemble_temporal_volumes(source,path,tmp_path/'result',max_memory_bytes=1)
    (source/'volume-3.json').write_text('{}')
    with pytest.raises(ValueError,match='hash changed'):assemble_temporal_volumes(source,path,tmp_path/'result')


def test_destination_preserved(tmp_path):
    source,path,_=fixture(tmp_path)
    out=tmp_path/'result';out.mkdir();(out/'keep.txt').write_text('existing')
    with pytest.raises(FileExistsError):assemble_temporal_volumes(source,path,out)
    assert (out/'keep.txt').read_text()=='existing'


def test_clock_format_is_explicit():
    assert time_key('173912.75')==time_key('17:39:12.750000')
    with pytest.raises(ValueError):time_key('series-017')


def test_trigger_partition_requires_explicit_verified_mapping(tmp_path):
    source,path,audit=fixture(tmp_path)
    for record in audit['records']:
        h=record['header'];h['TriggerTime']=(h['TemporalPositionIdentifier']-1)*2000
    path.write_text(json.dumps(audit))
    conversion=json.loads((source/'conversion.json').read_text())
    for row in conversion['outputs']:
        js=source/row['sidecar'];meta=json.loads(js.read_text())
        trigger=time_key(meta['AcquisitionTime'])/1000-time_key('120000')/1000
        if trigger:meta['TriggerDelayTime']=trigger
        meta['BidsGuess']=['func',row['nifti']]
        js.write_text(json.dumps(meta));row['sidecar_sha256']=sha256(js)
    (source/'conversion.json').write_text(json.dumps(conversion))
    with pytest.raises(ValueError,match='metadata differences'):
        assemble_temporal_volumes(source,path,tmp_path/'default')
    assemble_temporal_volumes(source,path,tmp_path/'verified',allow_verified_trigger_time_partition=True)
    result=json.loads((tmp_path/'verified/assembled.json').read_text())
    assert 'TriggerDelayTime' not in result and 'BidsGuess' not in result
    saved=json.loads((tmp_path/'verified/assembly.json').read_text())
    assert [r['dicom_trigger_time_ms'] for r in saved['ordered_inputs']]==[0,2000,4000]
    audit['records'][0]['header']['TriggerTime']=999
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='Trigger partition|trigger field'):
        assemble_temporal_volumes(source,path,tmp_path/'bad',allow_verified_trigger_time_partition=True)


def test_nontrivial_physical_scaling_preserved(tmp_path):
    source,path,_=fixture(tmp_path)
    conversion=json.loads((source/'conversion.json').read_text())
    for row in conversion['outputs']:
        p=source/row['nifti'];img=nib.load(p)
        data=np.asanyarray(img.dataobj).copy()
        changed=nib.Nifti1Image(data,img.affine,img.header)
        changed.header.set_slope_inter(.125,-3.25)
        nib.save(changed,p)
        row['nifti_sha256']=sha256(p)
    (source/'conversion.json').write_text(json.dumps(conversion))
    assemble_temporal_volumes(source,path,tmp_path/'result')
    np.testing.assert_array_equal(nib.load(tmp_path/'result/assembled.nii.gz').get_fdata()[0,0,0],[-3.125,-3,-2.875])
