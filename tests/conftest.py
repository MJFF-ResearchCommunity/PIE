"""Test-run plumbing, no fixtures: the `slow` marker, and skips for what the environment lacks.

CI subset: pytest -m "not slow and not ppmi"
"""
import functools
import importlib.util
import shutil

import pytest

# pie-clean (requirements.txt, git main) is outside the imaging environment, and these modules import it at the top
if importlib.util.find_spec("pie_clean") is None:
    collect_ignore = ["test_data_loader.py", "test_data_reducer.py", "test_from_fs.py", "test_pie_clean.py",
                      "test_pipeline.py"]

# Over 5 s each, timed one file at a time on the dev laptop at 2 threads (11 Oct 2026); everything else is under 5 s.
# A node id without [params] marks every parametrisation.
SLOW = {
    "tests/test_classifier.py::test_gbdt_comparison_finishes_and_honours_include",                 # 42 s, GBDT fits
    "tests/test_cnn.py::test_network_and_cv",                                                      # 12 s, CNN training
    "tests/test_embed.py::test_forward_shapes_random_weights",                                     # 14 s, 3 backbones
    "tests/test_embed.py::test_to_mni_shape_and_masking",                                          # 6 s, registration
    "tests/test_stats_small_sample.py::test_null_of_the_naive_design_sits_far_above_chance",       # 12 s, permutations
    "tests/test_stats_small_sample.py::test_naive_subset_search_is_optimistic_on_noise_and_nested_is_not",  # 7 s
}

# Tests that need a module or the dcm2niix binary the environment may not have (keyed by node id or module path)
NEEDS = {
    "tests/test_nm.py::test_a_subjects_series_read_the_zip_index_once": "dcm2niix",
    "tests/test_stats_survive.py": "lifelines",          # pandas<3: general PIE env only, not requirements-imaging.lock
    "tests/test_stats_correlate.py::test_partial_correlation_removes_confounder": "pingouin",
    "tests/test_stats_compare.py::test_dunn_posthoc_returns_pairwise": "scikit_posthocs",
}


@functools.cache
def _available(need):
    if need == "dcm2niix":
        from pie.imaging.convert import DCM2NIIX     # $PIE_DCM2NIIX, else <repo>/venv_imaging/bin/dcm2niix
        return shutil.which(DCM2NIIX) is not None
    return importlib.util.find_spec(need) is not None


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = item.nodeid.split("[")[0]
        if name in SLOW:
            item.add_marker(pytest.mark.slow)
        need = NEEDS.get(name) or NEEDS.get(name.split("::")[0])
        if need and not _available(need):
            item.add_marker(pytest.mark.skip(reason=f"needs {need}"))
