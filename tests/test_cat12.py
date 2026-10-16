"""pie.imaging.cat12 on synthetic inputs: the batch overrides, the Ch4 sum on modulated GM, the report parsing."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.imaging.cat12 import OVERRIDES, batch_text, ch4_from_mwp1, report_values


def test_batch_overrides_hit_one_line_each_and_fail_loudly_when_the_release_changes(tmp_path):
    pre = "matlabbatch{1}.spm.tools.cat.estwrite."
    shipped = "".join(f"{pre}{k} = 1;   % comment\n" for k in OVERRIDES) + f"{pre}output.GM.mod = 1;\n"
    (tmp_path / "b.m").write_text(shipped)
    text = batch_text(tmp_path / "b.m")
    assert f"{pre}output.surface = 0;\n" in text and f"{pre}output.warps = [0 0];\n" in text and f"{pre}output.GM.mod = 1;" in text
    (tmp_path / "b.m").write_text(shipped.replace(f"{pre}output.surface = 1;", ""))
    with pytest.raises(ValueError, match="output.surface"):
        batch_text(tmp_path / "b.m")


def test_ch4_sums_modulated_grey_matter_in_the_map_on_a_coarser_grid(tmp_path):
    aff1 = np.diag([1.0, 1, 1, 1])
    p = np.zeros((12, 12, 12), np.float32)
    p[2:8, 2:8, 2:8] = 0.8                     # 216 mm^3 at p >= 0.5
    nib.save(nib.Nifti1Image(p, aff1), tmp_path / "lh.nii.gz")
    aff2 = np.diag([2.0, 2, 2, 1])             # mwp1 on a 2 mm grid, values = native mm^3 per template mm^3
    gm = np.full((6, 6, 6), 0.5, np.float32)
    nib.save(nib.Nifti1Image(gm, aff2), tmp_path / "mwp1.nii")
    out = ch4_from_mwp1(tmp_path / "mwp1.nii", {"lh": tmp_path / "lh.nii.gz"})
    assert out["ch4_lh_gm_mm3"] == pytest.approx(0.5 * 216, rel=0.35)
    assert out["ch4_lh_pgm_mm3"] < out["ch4_lh_gm_mm3"]


def test_report_values_converts_units_and_reads_the_basal_forebrain(tmp_path):
    (tmp_path / "cat.xml").write_text("<S><qualityratings><IQR>2.0</IQR></qualityratings>"
                                      "<subjectmeasures><vol_TIV>1500.0</vol_TIV></subjectmeasures></S>")
    (tmp_path / "roi.xml").write_text("<S><neuromorphometrics><names><item>Left Basal Forebrain</item>"
                                      "<item>Right Basal Forebrain</item><item>Left Caudate</item></names><data><Vgm>[0.40;0.38;3.1]</Vgm></data></neuromorphometrics></S>")
    v = report_values(tmp_path / "cat.xml", tmp_path / "roi.xml")
    assert v["cat12_tiv_mm3"] == 1.5e6 and v["cat12_iqr_pct"] == 85
    assert v["nmm_lh_basal_forebrain_mm3"] == pytest.approx(400) and v["nmm_rh_basal_forebrain_mm3"] == pytest.approx(380)


def test_output_voxel_size_is_an_option_that_hits_one_line(tmp_path):
    pre = "matlabbatch{1}.spm.tools.cat.estwrite."
    shipped = "".join(f"{pre}{k} = 1;\n" for k in OVERRIDES) + f"{pre}extopts.registration.vox = 1.5;\n"
    (tmp_path / "b.m").write_text(shipped)
    assert f"{pre}extopts.registration.vox = 1;\n" in batch_text(tmp_path / "b.m", vox=1)
    assert f"{pre}extopts.registration.vox = 1.5;\n" in batch_text(tmp_path / "b.m", vox=1.5)


def test_configurations_keep_the_15_oct_result_reproducible():
    from pie.imaging import cat12
    legacy = cat12.CONFIGS["julich3-1.5mm"]
    assert (legacy["map"], legacy["vox"]) == ("julich3", 1.5)
    assert cat12.result_name("julich3-1.5mm") == "cat12_ch4.json"          # the file the frozen 13 Oct script reads
    assert cat12.work_name(1.5) == "cat12" and cat12.work_name(1) == "cat12_1mm"
    assert cat12.DEFAULT_CONFIG in cat12.CONFIGS and cat12.CONFIGS[cat12.DEFAULT_CONFIG].get("why_default")
    for name in cat12.CONFIGS:
        if name != "julich3-1.5mm":
            assert cat12.result_name(name) == f"cat12_ch4_{name}.json"


def test_unknown_ch4_map_is_refused_with_the_options():
    from pie.imaging.atlases import CH4_MAPS, ch4_maps
    assert "julich3" in CH4_MAPS
    with pytest.raises(ValueError, match="julich3"):
        ch4_maps("no-such-map")
