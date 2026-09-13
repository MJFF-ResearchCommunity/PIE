"""Synthetic-only tests for the cohort rules; no PPMI tables are read."""
import numpy as np
import pandas as pd
import pytest

from pie.experiment import cohort


def test_unknown_genotypes_are_not_controls():
    d = pd.DataFrame([[0, 0, 0], [0, np.nan, 0], [1, np.nan, np.nan], [np.nan] * 3], columns=cohort.GENES)
    out = cohort.carrier_status(d)
    assert out[0] == 0 and out[2] == 1 and out.loc[[1, 3]].isna().all()


def test_concurrent_visit_is_not_a_followup_label():
    d = pd.DataFrame({"T1_EVENT_ID": ["BL", "SC", "BL", "UNK", "V04"],
                      "SAA_EVENT_ID": ["BL", "BL", "V04", "UNK", "V04"]})
    assert cohort.concurrent_visit(d, "T1_EVENT_ID", "SAA_EVENT_ID").tolist() == [True, True, False, False, True]


def test_concurrent_visit_without_the_columns_is_false_not_an_error():
    assert not cohort.concurrent_visit(pd.DataFrame(index=range(3)), "a", "b").any()


def test_complete_case_excludes_infinities_and_missing_values():
    d = pd.DataFrame({"nm": [1., np.nan, 2., np.inf], "fw": [1., 2., np.nan, 1.]})
    assert cohort.complete_case_mask(d, ["nm", "fw"]).tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        cohort.complete_case_mask(d, [])


def test_sex_is_decoded_per_module_and_an_unknown_module_raises():
    d = pd.DataFrame({"PAG_NAME": ["SCREEN", "SCREEN", "PARTICIPANT_PROFILE", "PARTICIPANT_PROFILE"],
                      "SEX": [1, 0, 1, 2]})
    assert cohort.decode_sex(d).tolist() == [1., 0., 1., 0.]
    with pytest.raises(ValueError, match="Unverified demographic module"):
        cohort.decode_sex(pd.DataFrame({"PAG_NAME": ["SOME_NEW_EXPORT"], "SEX": [1]}))


def test_conflicting_repeats_are_dropped_rather_than_resolved_by_row_order():
    d = pd.DataFrame({"PATNO": [1, 1, 2, 2], "SEX": [1, 0, 1, 1]})
    out = cohort.unique_per_participant(d, "SEX")
    assert out.index.tolist() == [2] and out.loc[2] == 1
