from typing import Any, Dict, List, Type, Union

from pydantic import BaseModel

from phd_model_evaluations.data.feature.checker.gap_with_punctuation_feature import GapWithPunctuationFeatureChecker
from phd_model_evaluations.data.feature.checker.is_number_feature import IsNumberFeatureChecker
from phd_model_evaluations.data.feature.checker.left_context_length_feature import LeftContextLengthFeatureChecker
from phd_model_evaluations.data.feature.checker.masked_token_frequency_feature import MaskedTokenFrequencyFeatureChecker
from phd_model_evaluations.data.feature.checker.masked_token_length_feature import MaskedTokenLengthFeatureChecker
from phd_model_evaluations.data.feature.checker.right_context_length_feature import RightContextLengthFeatureChecker
from phd_model_evaluations.data.feature.checker.text_length_feature import TextLengthFeatureChecker

ALL_FEATURE_CHECKERS_TYPE = Union[
    TextLengthFeatureChecker,
    MaskedTokenLengthFeatureChecker,
    LeftContextLengthFeatureChecker,
    RightContextLengthFeatureChecker,
    GapWithPunctuationFeatureChecker,
    IsNumberFeatureChecker,
    MaskedTokenFrequencyFeatureChecker,
]

ALL_FEATURE_CHECKERS_CLASSES: List[Type[ALL_FEATURE_CHECKERS_TYPE]] = [
    TextLengthFeatureChecker,
    MaskedTokenLengthFeatureChecker,
    LeftContextLengthFeatureChecker,
    RightContextLengthFeatureChecker,
    GapWithPunctuationFeatureChecker,
    IsNumberFeatureChecker,
    MaskedTokenFrequencyFeatureChecker,
]


class LMGapFeatureCheckers(BaseModel):
    checkers: List[ALL_FEATURE_CHECKERS_TYPE]


def parse_single_lm_gap_feature_checkers(data: Dict[str, Any]) -> ALL_FEATURE_CHECKERS_TYPE:
    name = data["name"]
    for checker_cls in ALL_FEATURE_CHECKERS_CLASSES:
        if checker_cls.get_checker_name() == name:
            return checker_cls(**data)

    raise RuntimeError(f"Cannot create LM-GAP feature checker with name: {name}")


def parse_lm_gap_feature_checkers(data: Dict[str, Any]) -> LMGapFeatureCheckers:
    return LMGapFeatureCheckers(
        checkers=[parse_single_lm_gap_feature_checkers(single_data) for single_data in data["checkers"]]
    )


def get_example_lm_gap_feature_checkers() -> LMGapFeatureCheckers:
    return LMGapFeatureCheckers(
        checkers=[
            TextLengthFeatureChecker(invert_checking=False, min_text_length=9, max_text_length=25),
            MaskedTokenLengthFeatureChecker(
                invert_checking=False, min_masked_token_length=4, max_masked_token_length=9
            ),
            LeftContextLengthFeatureChecker(
                invert_checking=False, min_left_context_length=3, max_left_context_length=50
            ),
            RightContextLengthFeatureChecker(
                invert_checking=False, min_right_context_length=3, max_right_context_length=50
            ),
            GapWithPunctuationFeatureChecker(
                invert_checking=True,
                check_start_with_punctuation=True,
                check_end_with_punctuation=True,
                check_has_punctuation=False,
                check_is_punctuation=False,
            ),
            IsNumberFeatureChecker(invert_checking=True, check_number=True),
            MaskedTokenFrequencyFeatureChecker(
                invert_checking=False, min_masked_token_frequency=3, max_masked_token_frequency=1_000_000
            ),
        ]
    )
