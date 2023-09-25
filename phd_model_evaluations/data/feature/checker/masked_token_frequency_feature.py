from typing import Dict, Union

from phd_model_evaluations.data.feature.checker.base_feature import BaseFeature, FeatureChecker
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType

FEATURE_NAME = "masked_token_frequency"


class MaskedTokenFrequencyFeature(BaseFeature):
    name: str = FEATURE_NAME
    masked_token: str
    masked_token_frequency: int

    def get_aggregation_data(self) -> Dict[str, int]:
        return {self.masked_token: self.masked_token_frequency}

    @staticmethod
    def get_draw_option() -> DrawOptionFeature:
        return DrawOptionFeature(
            visualization_type=VisualizationType.pie,
            title_x="-",
            title_y="-",
            other_label_name="Pozostałe",
            map_key_to_label_name={},
        )


class MaskedTokenFrequencyFeatureChecker(FeatureChecker):
    name: str = FEATURE_NAME
    min_masked_token_frequency: int
    max_masked_token_frequency: int

    def _is_accepted_value(
        self, feature: Union[BaseFeature, MaskedTokenFrequencyFeature], statistics: FeatureStatistics
    ) -> bool:
        feature = self.check_expected_type(MaskedTokenFrequencyFeature, feature)
        return (
            self.min_masked_token_frequency
            <= statistics.token_frequency.get(feature.masked_token, 0)
            <= self.max_masked_token_frequency
        )

    @staticmethod
    def get_checker_name() -> str:
        return FEATURE_NAME
