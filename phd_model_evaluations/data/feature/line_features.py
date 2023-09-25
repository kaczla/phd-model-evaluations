from typing import List, Union

from pydantic import BaseModel

from phd_model_evaluations.data.feature.checker.gap_with_punctuation_feature import GapWithPunctuationFeature
from phd_model_evaluations.data.feature.checker.is_number_feature import IsNumberFeature
from phd_model_evaluations.data.feature.checker.left_context_length_feature import LeftContextLengthFeature
from phd_model_evaluations.data.feature.checker.masked_token_frequency_feature import MaskedTokenFrequencyFeature
from phd_model_evaluations.data.feature.checker.masked_token_length_feature import MaskedTokenLengthFeature
from phd_model_evaluations.data.feature.checker.right_context_length_feature import RightContextLengthFeature
from phd_model_evaluations.data.feature.checker.text_length_feature import TextLengthFeature

ALL_FEATURES_TYPE = Union[
    TextLengthFeature,
    MaskedTokenLengthFeature,
    LeftContextLengthFeature,
    RightContextLengthFeature,
    GapWithPunctuationFeature,
    IsNumberFeature,
    MaskedTokenFrequencyFeature,
]


class LineFeatures(BaseModel):
    line_id: int
    features: List[ALL_FEATURES_TYPE]
