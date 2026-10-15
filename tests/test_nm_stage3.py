import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nm_stage3  # noqa: E402


def test_stage3_pass_needs_both_halves_of_the_rule():
    assert nm_stage3.stage3_pass([0.55, 0.70])
    assert not nm_stage3.stage3_pass([0.49, 0.70])          # chance not excluded
    assert not nm_stage3.stage3_pass([0.51, 0.62])          # shown worse than the published 0.633


def test_restrict_counts_participants_used_before():
    d = pd.DataFrame({"patno": [1, 2, 3, 4]})
    out, flow = nm_stage3.restrict(d, {}, fresh={1, 2, 3}, exclude={2})
    assert out["patno"].tolist() == [1, 3]
    assert flow == {"not_in_fresh_lists": 1, "excluded_used_before": 1}


def test_dry_runs_use_fake_data_only(tmp_path):
    for analysis in ("fresh-clinical", "prodromal-fresh"):
        res = nm_stage3.main([analysis, "--synthetic", "--out", str(tmp_path)])
        assert set(res["secondary"]["descriptive"]) == set(nm_stage3.SECONDARY)
        assert "pass_rule" in res
    assert json.loads((tmp_path / "synthetic_fresh_clinical_result.json").read_text())["flow"]["synthetic"]
