import nibabel as nib
import numpy as np
import pytest

from pie.imaging.fmri_qc import spatial_montage


def test_comparison_renders_both_layers_without_changing_source(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from pie.imaging.fmri_qc import render_report_states
    svg = tmp_path / 'comparison.svg'
    content = '<svg xmlns="http://www.w3.org/2000/svg"><g class="background-svg"><text>before</text></g><g class="foreground-svg"><text>after</text></g></svg>'
    svg.write_text(content)
    captured = []
    monkeypatch.setitem(sys.modules, 'cairosvg', SimpleNamespace(svg2png=lambda **kw: captured.append(kw)))
    paths = render_report_states(svg, tmp_path / 'review')
    assert len(paths) == 2
    import xml.etree.ElementTree as ET
    for index, selected in enumerate(('background-svg', 'foreground-svg')):
        nodes = {node.get('class'): node for node in ET.fromstring(captured[index]['bytestring']).iter() if node.get('class')}
        assert len(nodes) == 2  # Retain shared resources even in hidden layers.
        assert 'display:inline' in nodes[selected].get('style')
        assert 'display:none' in nodes[({'background-svg','foreground-svg'} - {selected}).pop()].get('style')
    assert svg.read_text() == content


def test_montage_explicit_world_grids(tmp_path):
    data = np.ones((8, 9, 10), dtype=np.float32)
    mask = np.zeros_like(data)
    mask[2:6, 2:7, 2:8] = 1
    image, overlay = tmp_path / 'image.nii.gz', tmp_path / 'overlay.nii.gz'
    affine = np.diag([-2., 2., 3., 1.])
    nib.save(nib.Nifti1Image(data, affine), image)
    nib.save(nib.Nifti1Image(mask, affine), overlay)
    output = tmp_path / 'review.png'
    assert spatial_montage(image, {'mask': (overlay, .5, 'lime')}, output, mask=overlay) == str(output)
    assert output.stat().st_size > 1000
    nib.save(nib.Nifti1Image(np.zeros_like(data), affine), overlay)
    with pytest.raises(ValueError, match='Empty montage'):
        spatial_montage(image, {}, output, mask=overlay)


def test_montage_titles_give_world_position_of_oblique_slice_centre(tmp_path, monkeypatch):
    import matplotlib.axes
    titles = []
    monkeypatch.setattr(matplotlib.axes.Axes, 'set_title', lambda self, text, **kw: titles.append(text))
    angle = np.deg2rad(30)   # rotation about x: axial voxel slices tilt in world z
    affine = np.array([[2, 0, 0, -20], [0, 2 * np.cos(angle), -2 * np.sin(angle), -10],
                       [0, 2 * np.sin(angle), 2 * np.cos(angle), 5], [0, 0, 0, 1]])
    image = tmp_path / 'oblique.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((20, 20, 20), np.float32), affine), image)
    spatial_montage(image, {}, tmp_path / 'review.png')
    canonical = nib.as_closest_canonical(nib.load(image)).affine
    # Middle axial cut is voxel k=10; the displayed support centre is (9.5, 9.5).
    expected = nib.affines.apply_affine(canonical, [9.5, 9.5, 10])[2]
    assert titles[2] == f'z = {expected:.0f} mm'
