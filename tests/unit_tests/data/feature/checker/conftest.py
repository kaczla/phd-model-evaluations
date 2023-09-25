from collections import Counter

import pytest

from phd_model_evaluations.data.feature.checker.masked_token_frequency_feature import (
    MaskedTokenFrequencyFeature,
    MaskedTokenFrequencyFeatureChecker,
)
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics


@pytest.fixture()
def feature_statistics() -> FeatureStatistics:
    return FeatureStatistics(
        token_frequency=Counter({"new": 15, "the": 20}),
        sequence_token_length=Counter({1: 2, 2: 1, 4: 4, 5: 10, 6: 11, 7: 12}),
        are_aggregated=True,
    )


@pytest.fixture()
def masked_token_frequency_feature() -> MaskedTokenFrequencyFeature:
    return MaskedTokenFrequencyFeature(masked_token="new", masked_token_frequency=45)


@pytest.fixture()
def masked_token_frequency_feature_checker() -> MaskedTokenFrequencyFeatureChecker:
    return MaskedTokenFrequencyFeatureChecker(
        invert_checking=False, min_masked_token_frequency=3, max_masked_token_frequency=96
    )
