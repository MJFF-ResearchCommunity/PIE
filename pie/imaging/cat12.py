"""Nucleus basalis (Ch4) grey-matter volume with CAT12, the method of the PD, PPMI and SAA Ch4 literature (Grothe et al.
2012 Cereb Cortex; Ray et al. 2018 Brain, on PPMI; Wisse, Spotorno et al. 2024): CAT12 segments the raw T1 and
registers it to its MNI152NLin2009cAsym Shooting template; grey matter modulated for the warp (``mwp1``: native mm^3
per voxel) is summed inside the Julich-Brain Ch4 maps (Zaborszky et al. 2008; ``atlases.ch4_maps``) at p >= 0.5, and
weighted by probability. CAT12's TIV gives head size and its IQR the image quality. The neuromorphometrics atlas's
basal forebrain (CAT12's own ROI output) is kept as a second definition.

This replaces FSL FAST grey matter in the SyN-pulled maps, which failed plausibility on PPMI (no age effect in
controls, 26 % of the map counted as grey matter; documentation/imaging.md).

Needs the CAT12 standalone and the free MATLAB Runtime R2023b: ``PIE_CAT12`` = the CAT12.9_R2023b_MCR_Linux folder
(holding standalone/cat_standalone.sh), ``PIE_MCR`` = the Runtime folder holding R2023b's bin/ (default: under
~/cat12). Pinned: CAT12.9 r2665 standalone (compiled SPM25; R2023b build of 27 Mar 2025,
www.neuro.uni-jena.de/cat12/) on MATLAB Runtime R2023b Update 10."""
import gzip
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np

CAT12 = Path(os.environ.get("PIE_CAT12", Path.home() / "cat12" / "CAT12.9_R2023b_MCR_Linux"))
MCR = Path(os.environ.get("PIE_MCR", Path.home() / "cat12" / "MCR" / "R2023b"))
# A Ch4 configuration is a probability map (atlases.CH4_MAPS) and CAT12's output voxel size. Every result records its
# configuration; the default is the best-evidenced one, with the reason next to it. Keep old entries reproducible.
CONFIGS = {
    "julich3-1.5mm": {"map": "julich3", "vox": 1.5,
                      "status": "pre-registered 13 Oct 2026; not validated 15 Oct (controls' age r -0.299, rho with ScLimbic "
                                "0.10, ICC 0.76); its map is asymmetric (left/right 2.0)",
                      "why_default": "the only configuration so far"},
}
DEFAULT_CONFIG = "julich3-1.5mm"


def result_name(config):
    """stats/ file of a configuration's result; the 13 Oct configuration keeps the name its frozen script reads."""
    return "cat12_ch4.json" if config == "julich3-1.5mm" else f"cat12_ch4_{config}.json"


def work_name(vox):
    """CAT12's working folder for an output voxel size; any map at that size reuses its modulated grey matter."""
    return "cat12" if vox == 1.5 else f"cat12_{vox:g}mm"
# Changes to CAT12's shipped default batch (standalone/cat_standalone_segment.m), all outputs we do not use: no
# surfaces (most of the run time), only modulated GM, only the neuromorphometrics ROIs, no report figure.
OVERRIDES = {"output.surface": "0", "output.ROImenu.atlases.cobra": "0", "output.WM.mod": "0", "output.label.native": "0",
             "output.bias.warped": "0", "output.warps": "[0 0]", "extopts.admin.print": "0"}


def batch_text(shipped, vox=None):
    """CAT12's default segmentation batch with ``OVERRIDES`` (and the output voxel size, when given) applied; every
    override must hit exactly one line."""
    text = Path(shipped).read_text()
    overrides = OVERRIDES if vox is None else {**OVERRIDES, "extopts.registration.vox": f"{vox:g}"}
    for key, val in overrides.items():
        pat = re.compile(rf"^(matlabbatch\{{1\}}\.spm\.tools\.cat\.estwrite\.{re.escape(key)}\s*=).*$", re.M)
        text, n = pat.subn(rf"\g<1> {val};", text)
        if n != 1:
            raise ValueError(f"CAT12 batch has {n} lines for {key}: the pinned release changed")
    return text


def ch4_from_mwp1(mwp1, maps):
    """{"ch4_{h}_gm_mm3": modulated GM in the map at p >= 0.5, "ch4_{h}_pgm_mm3": weighted by p} for h in ``maps``."""
    from nibabel.processing import resample_from_to

    img = nib.load(mwp1)
    gm = np.nan_to_num(np.asarray(img.dataobj, np.float32))
    vox = float(abs(np.linalg.det(img.affine[:3, :3])))
    out = {}
    for h, path in maps.items():
        p = np.asarray(resample_from_to(nib.load(path), (gm.shape, img.affine), order=1).dataobj, np.float32)
        out[f"ch4_{h}_gm_mm3"] = float((gm * (p >= 0.5)).sum() * vox)
        out[f"ch4_{h}_pgm_mm3"] = float((gm * p).sum() * vox)
    return out


def _num(root, path):
    node = root.find(path)
    return float(node.text.strip().strip("[]").split()[0]) if node is not None and node.text else float("nan")


def report_values(report_xml, roi_xml=None):
    """CAT12's TIV (mm^3), image-quality rating (IQR, %) and, from the neuromorphometrics ROI file, the basal forebrain
    grey matter (mm^3) per side."""
    r = ET.parse(report_xml).getroot()
    mark = _num(r, "qualityratings/IQR")
    out = {"cat12_tiv_mm3": 1000 * _num(r, "subjectmeasures/vol_TIV"),       # CAT12 reports cm^3
           "cat12_iqr_pct": min(100.0, max(0.0, 105 - 10 * mark))}              # CAT12's mark2rps: mark 1 (best) -> 95 %
    if roi_xml and Path(roi_xml).exists():
        n = ET.parse(roi_xml).getroot().find("neuromorphometrics")
        names = [s.strip() for s in n.find("names").itertext() if s.strip()]
        vgm = [float(v) for v in n.find("data/Vgm").text.strip("[] \n").split(";")]
        for name, v in zip(names, vgm):                                        # CAT12.9 names, cm^3
            if name in ("Left Basal Forebrain", "Right Basal Forebrain"):
                out[f"nmm_{'lh' if name[0] == 'L' else 'rh'}_basal_forebrain_mm3"] = 1000 * v
    return out


def ch4_volume(subjects_dir, sid, nii, threads=1, config=None):
    """Ch4 volume of one configuration (``CONFIGS``; default ``PIE_CH4_CONFIG`` or ``DEFAULT_CONFIG``) into
    ``stats/<result_name>``: the keys of ``ch4_from_mwp1`` and ``report_values`` plus the configuration. CAT12 runs on
    the raw T1 (``nii``) into <subj>/<work_name> unless a finished run at that voxel size is there to reuse. Kept: the
    modulated GM (mwp1, gzipped) and CAT12's report and ROI XML."""
    from .atlases import ch4_maps

    config = config or os.environ.get("PIE_CH4_CONFIG", DEFAULT_CONFIG)
    if config not in CONFIGS:
        raise ValueError(f"unknown Ch4 configuration {config!r}; options: {', '.join(CONFIGS)}")
    cfg = CONFIGS[config]
    subj = Path(subjects_dir).resolve() / sid
    out = subj / "stats" / result_name(config)
    if out.exists():
        return json.loads(out.read_text())
    work = subj / work_name(cfg["vox"])
    done = work / "mwp1t1.nii.gz", work / "report" / "cat_t1.xml", work / "label" / "catROI_t1.xml"
    if all(p.exists() for p in done):                # this voxel size already ran: only the map differs
        row = {**ch4_from_mwp1(done[0], ch4_maps(cfg["map"])), **report_values(done[1], done[2]),
               "cat12_ch4_config": config, "cat12_ch4_map": cfg["map"], "cat12_vox_mm": cfg["vox"]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(row, indent=1))
        return row
    if not nii or not Path(nii).exists():
        raise FileNotFoundError(f"raw T1 missing: {nii}")
    shutil.rmtree(work, ignore_errors=True)          # a failed attempt's outputs must not pass for this one's
    work.mkdir(parents=True)
    t1 = work / "t1.nii"
    nib.save(nib.load(nii), t1)
    (work / "batch.m").write_text(batch_text(CAT12 / "standalone" / "cat_standalone_segment.m", vox=cfg["vox"]))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    r = subprocess.run(["bash", str(CAT12 / "standalone" / "cat_standalone.sh"), "-m", str(MCR), "-b", str(work / "batch.m"), str(t1)],
                       cwd=work, env=env, capture_output=True, text=True, timeout=4 * 3600)
    (work / "cat12.log").write_text(r.stdout[-20000:] + r.stderr[-5000:])
    mwp1, report = work / "mri" / "mwp1t1.nii", work / "report" / "cat_t1.xml"
    if r.returncode or not mwp1.exists() or not report.exists():
        raise RuntimeError(f"CAT12 failed (exit {r.returncode}): {(r.stderr or r.stdout)[-300:]}")
    row = {**ch4_from_mwp1(mwp1, ch4_maps(cfg["map"])), **report_values(report, work / "label" / "catROI_t1.xml"),
           "cat12_ch4_config": config, "cat12_ch4_map": cfg["map"], "cat12_vox_mm": cfg["vox"]}
    with open(mwp1, "rb") as f, gzip.open(work / "mwp1t1.nii.gz", "wb") as g:
        shutil.copyfileobj(f, g)
    for p in work.iterdir():                         # keep mwp1t1.nii.gz, the report/label XML and the log
        if p.is_dir() and p.name not in ("report", "label"):
            shutil.rmtree(p)
        elif p.suffix == ".nii":
            p.unlink()
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    return row
