"""Run from the repo root: PYTHONPATH=. venv_imaging/bin/python scripts/ch4_v2_map_only.py

Pre-registered descriptive (preregistration_ch4_v2.md): julich303-1.5mm on the 13 Oct scans, i.e. the 13 Oct CAT12
runs with only the map changed. Median left/right ratio and median volume over the 13 Oct analysed set (HC + PD,
IQR >= 60, covariates present). Prints aggregates only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import ch4_cat12_validation as v1
from pie.imaging.cat12 import ch4_volume

OUT = Path.home() / "pie_validation/20261013_ch4_cat12"
s = pd.read_csv(OUT / "sids.csv")
ids = s.loc[s["role"].isin(["hc", "pd"]), "IMAGEID"].to_numpy()
main, _ = v1.load("PPMI", OUT)
assert len(main) == len(ids)
rows, fails = [], 0
for i in ids:
    try:
        j = ch4_volume(v1.D / "fastsurfer", str(i), None, config="julich303-1.5mm")
        rows.append({k: j[k] for k in ("ch4_lh_gm_mm3", "ch4_rh_gm_mm3", "ch4_lh_pgm_mm3", "ch4_rh_pgm_mm3", "cat12_iqr_pct")})
    except Exception:
        rows.append({}); fails += 1
new = pd.DataFrame(rows, columns=["ch4_lh_gm_mm3", "ch4_rh_gm_mm3", "ch4_lh_pgm_mm3", "ch4_rh_pgm_mm3", "cat12_iqr_pct"])
same = np.isclose(new["cat12_iqr_pct"], main["iqr"], equal_nan=True).all()            # same CAT12 run, same row order
keep = ((main["iqr"] >= v1.IQR_MIN) & main[["ch4", "tiv", "age", "SEX", "bf", "eTIV"]].notna().all(axis=1)).to_numpy()
n, k = new[keep], main[keep]
res = {"config": "julich303-1.5mm", "scans": int(len(ids)), "failed": fails, "rows_aligned": bool(same), "n_analysed": int(keep.sum()),
       "gm_p05": {"lh_rh_median": float(np.nanmedian(n["ch4_lh_gm_mm3"] / n["ch4_rh_gm_mm3"])),
                  "median_mm3": float(np.nanmedian(n["ch4_lh_gm_mm3"] + n["ch4_rh_gm_mm3"]))},
       "prob_weighted": {"lh_rh_median": float(np.nanmedian(n["ch4_lh_pgm_mm3"] / n["ch4_rh_pgm_mm3"])),
                         "median_mm3": float(np.nanmedian(n["ch4_lh_pgm_mm3"] + n["ch4_rh_pgm_mm3"]))},
       "julich31_same_set": {"gm_p05_lh_rh_median": float(np.nanmedian(k["ch4_lh"] / k["ch4_rh"])), "gm_p05_median_mm3": float(np.nanmedian(k["ch4"])),
                             "prob_weighted_median_mm3": float(np.nanmedian(k["ch4_p"]))}}
(Path.home() / "pie_validation/20261016_ch4_v2/map_only_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
