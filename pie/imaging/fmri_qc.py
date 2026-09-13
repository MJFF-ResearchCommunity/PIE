"""Explicit-grid visual review of preprocessed fMRI and anatomical derivatives."""
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def spatial_montage(background, overlays, output, *, title='', mask=None):
    """Render five cuts in each orthogonal plane, using physical-space overlays.

    ``overlays`` maps legend labels to (image path, contour level, color).
    No registration is estimated; inputs must already share physical coordinates.
    This image is for human review and never assigns an automatic QC pass.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    img = nib.as_closest_canonical(nib.load(background))
    if img.ndim != 3:
        raise ValueError('Montage background must be 3D')
    data = img.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite montage background')
    arrays = {}
    for label, (path, level, color) in overlays.items():
        overlay = resample_from_to(nib.load(path), img, order=1).get_fdata()
        if not np.isfinite(overlay).all():
            raise ValueError('Nonfinite montage overlay')
        arrays[label] = (overlay, level, color)
    support = data > 0 if mask is None else resample_from_to(nib.load(mask), img, order=0).get_fdata() > 0
    points = np.argwhere(support)
    if not len(points):
        raise ValueError('Empty montage support')
    low, high = points.min(axis=0), points.max(axis=0)
    spacing = img.header.get_zooms()
    vmax = np.percentile(data[support], 99)
    fig, axes = plt.subplots(3, 5, figsize=(15, 10), facecolor='black')
    for row, axis in enumerate((2, 1, 0)):
        for col, fraction in enumerate((.12, .3, .5, .7, .88)):
            index = int(round(low[axis] + fraction * (high[axis]-low[axis])))
            other = [i for i in range(3) if i != axis]
            ax = axes[row, col]
            ax.imshow(np.take(data, index, axis=axis).T, origin='lower', cmap='gray',
                      vmin=0, vmax=vmax, aspect=spacing[other[1]] / spacing[other[0]])
            for overlay, level, color in arrays.values():
                cut = np.take(overlay, index, axis=axis).T
                if cut.min() < level < cut.max():
                    ax.contour(cut, levels=[level], colors=[color], linewidths=.6)
            world = nib.affines.apply_affine(img.affine, np.eye(3)[axis] * index)[axis]
            ax.set_title(f'{"xyz"[axis]} = {world:.0f} mm', color='white', fontsize=9)
            ax.axis('off')
    fig.suptitle(title + ' | RAS coordinates; visual review required', color='white', fontsize=11)
    if arrays:
        fig.legend([Line2D([0], [0], color=color) for _, _, color in arrays.values()],
                   list(arrays), loc='lower center', ncol=len(arrays), facecolor='black', labelcolor='white')
    fig.tight_layout(rect=(0, .035, 1, .97))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=120, facecolor='black')
    plt.close(fig)
    return str(target)


def render_report_states(svg, output_dir, *, output_width=1600):
    """Render both layers of an animated niworkflows comparison, without edits.

    A single static SVG rendering hides one comparison image. Preserve the
    source and render foreground/background separately for auditable inspection.
    CairoSVG is an optional dependency needed only for this report renderer.
    """
    import copy
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import cairosvg

    source = Path(svg)
    root = ET.parse(source).getroot()
    classes = {node.get('class') for node in root.iter()}
    states = ('background', 'foreground') if {'background-svg', 'foreground-svg'} <= classes else ('static',)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for state in states:
        tree = copy.deepcopy(root)
        if state != 'static':
            for node in tree.iter():
                if node.get('class') in {'background-svg', 'foreground-svg'}:
                    # Keep hidden definitions: report compression may share SVG
                    # resources between layers. Removing a layer corrupts the
                    # remaining image in some valid niworkflows reports.
                    visible = node.get('class') == state + '-svg'
                    node.set('style', 'display:' + ('inline' if visible else 'none') + ';animation:none;opacity:1')
        target = output / (source.stem + '_' + state + '.png')
        cairosvg.svg2png(bytestring=ET.tostring(tree), write_to=str(target), output_width=output_width)
        paths.append(target)
    return paths
