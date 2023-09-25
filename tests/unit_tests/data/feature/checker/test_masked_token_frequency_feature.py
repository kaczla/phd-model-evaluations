from phd_model_evaluations.data.feature.checker.masked_token_frequency_feature import (
    MaskedTokenFrequencyFeature,
    MaskedTokenFrequencyFeatureChecker,
)
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType


def test_masked_token_frequency_feature(masked_token_frequency_feature: MaskedTokenFrequencyFeature) -> None:
    assert masked_token_frequency_feature.name == "masked_token_frequency", "Invalid name of LM-GAP feature"
    assert masked_token_frequency_feature.masked_token == "new", "Invalid masked token"
    assert masked_token_frequency_feature.masked_token_frequency == 45, "Invalid frequency of masked token"
    assert masked_token_frequency_feature.get_aggregation_data() == {"new": 45}, "Invalid aggregation data"
    expected_draw_option = DrawOptionFeature(
        visualization_type=VisualizationType.histogram,
        title_x="Częstość wystąpienia",
        title_y="Liczba wystąpień",
        other_label_name="-",
        map_key_to_label_name={},
    )
    assert masked_token_frequency_feature.get_draw_option() == expected_draw_option, "Invalid draw option"


def test_masked_token_frequency_feature_checker(
    masked_token_frequency_feature_checker: MaskedTokenFrequencyFeatureChecker,
) -> None:
    assert (
        masked_token_frequency_feature_checker.name == "masked_token_frequency"
    ), "Invalid name of LM-GAP feature checker"
    assert (
        masked_token_frequency_feature_checker.name == masked_token_frequency_feature_checker.get_checker_name()
    ), "Invalid name of LM-GAP feature checker"
    assert not masked_token_frequency_feature_checker.invert_checking, "Invalid invert checking flag"
    assert (
        masked_token_frequency_feature_checker.min_masked_token_frequency == 3
    ), "Invalid min frequency of masked token"
    assert (
        masked_token_frequency_feature_checker.max_masked_token_frequency == 96
    ), "Invalid max frequency of masked token"


def test_masked_token_frequency_feature_checker_success(
    feature_statistics: FeatureStatistics,
    masked_token_frequency_feature: MaskedTokenFrequencyFeature,
    masked_token_frequency_feature_checker: MaskedTokenFrequencyFeatureChecker,
) -> None:
    assert masked_token_frequency_feature_checker.is_accepted_value(
        masked_token_frequency_feature, feature_statistics
    ), "Invalid status of accepted value"


def test_masked_token_frequency_feature_checker_failed(
    feature_statistics: FeatureStatistics, masked_token_frequency_feature_checker: MaskedTokenFrequencyFeatureChecker
) -> None:
    masked_token_frequency_feature = MaskedTokenFrequencyFeature(masked_token="N E W", masked_token_frequency=1)
    assert not masked_token_frequency_feature_checker.is_accepted_value(
        masked_token_frequency_feature, feature_statistics
    ), "Invalid status of accepted value"
