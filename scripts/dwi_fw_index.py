"""DWI free-water study (documentation/preregistration_dwi_fw.md): the series index for ``dwi --eddy --study --per-visit``,
restricted to each participant's earliest three-shell session (PPMI 2.0 Siemens Prisma: DTI_B700/B1000/B2000_64dir
series, two or more shells on one date), with every series of that date kept (its b0s and reverse-PE b0s). Reads zip
member names only, never an image. ``--per-visit`` then measures the session on the nearest finished T1 within 180 days.

    PYTHONPATH=. python scripts/dwi_fw_index.py --zips <zip> [...] --out Imaging/derived/dwi_fw/dwi_index.csv
"""
import argparse
from pathlib import Path

SHELL = r"(?i)DTI_b(700|1000|2000)_64dir"


def earliest_three_shell(idx):
    """Rows of each participant's earliest dated session holding >= 2 of the three shells."""
    sh = idx["desc"].astype(str).str.extract(SHELL, expand=False)
    n = idx.assign(_sh=sh).dropna(subset=["_sh"]).groupby(["patno", "date"])["_sh"].nunique()
    first = n[n >= 2].reset_index().sort_values("date").drop_duplicates("patno")[["patno", "date"]]
    return idx.merge(first, on=["patno", "date"])


def main(argv=None):
    from pie.imaging.batch import index_series

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    idx = index_series(a.zips).drop_duplicates("image_id")
    out = earliest_three_shell(idx)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"{len(idx)} series indexed; {out['patno'].nunique()} participants with a three-shell session, {len(out)} series kept")


if __name__ == "__main__":
    main()
