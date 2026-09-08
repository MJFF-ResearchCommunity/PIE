"""
PIE imaging layer: raw PPMI MRI (LONI DICOM zips) -> NIfTI -> FastSurfer
segmentation -> per-visit imaging-derived phenotypes (IDPs) that join the
tabular PIE data on PATNO / EVENT_ID.

Modules
-------
index      : list series inside LONI zips and pick one T1w per session
convert    : extract a series and convert it with dcm2niix (keeps NIfTI + JSON sidecar)
link       : map an MRI session date to a PPMI EVENT_ID
fastsurfer : run FastSurfer segmentation-only and parse its stats files
features   : assemble the wide IDP table
labels     : DaT-deficit and SAA labels aligned to an MRI session
run        : resumable CLI that does all of the above
"""

from pie.imaging.index import index_zips, select_t1_series
from pie.imaging.convert import convert_series
from pie.imaging.link import link_sessions_to_events
from pie.imaging.fastsurfer import run_fastsurfer, parse_stats
from pie.imaging.features import build_idp_table
from pie.imaging.labels import dat_labels, saa_labels

__all__ = [
    "index_zips", "select_t1_series", "convert_series", "link_sessions_to_events",
    "run_fastsurfer", "parse_stats", "build_idp_table", "dat_labels", "saa_labels",
]
