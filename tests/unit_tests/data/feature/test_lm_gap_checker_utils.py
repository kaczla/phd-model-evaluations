import pytest

from phd_model_evaluations.data.feature.checker.is_number_feature import IsNumberFeatureChecker
from phd_model_evaluations.data.feature.lm_gap_checker_utils import (
    get_example_lm_gap_feature_checkers,
    parse_lm_gap_feature_checkers,
    parse_single_lm_gap_feature_checkers,
)


def test_parse_example_lm_gap_feature_checkers() -> None:
    example_lm_gap_feature_checkers = get_example_lm_gap_feature_checkers()
    result = parse_lm_gap_feature_checkers(example_lm_gap_feature_checkers.dict())
    assert example_lm_gap_feature_checkers == result, "Invalid result of parsing LM-GAP feature checkers"


def test_parse_single_lm_gap_feature_checkers() -> None:
    checker = parse_single_lm_gap_feature_checkers(
        {"name": "is_number", "check_number": True, "invert_checking": False}
    )
    assert isinstance(checker, IsNumberFeatureChecker), "Invalid parser LM-GAP feature checker"
    assert checker.name == "is_number", "Invalid name of LM-GAP feature checker"
    assert checker.check_number, "Invalid check number in IsNumberFeatureChecker"
    assert not checker.invert_checking, "Invalid invert checking in IsNumberFeatureChecker"


def test_parse_single_lm_gap_feature_checkers_exception() -> None:
    with pytest.raises(RuntimeError):
        parse_single_lm_gap_feature_checkers({"name": "some_checker_123"})
