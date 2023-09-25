from typing import Dict, Union

from phd_model_evaluations.data.feature.checker.base_feature import BaseFeature, FeatureChecker
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType

FEATURE_NAME = "left_context_length"


class LeftContextLengthFeature(BaseFeature):
    name: str = FEATURE_NAME
    left_context_length: int

    def get_aggregation_data(self) -> Dict[int, int]:
        return {self.left_context_length: 1}

    @staticmethod
    def get_draw_option() -> DrawOptionFeature:
        return DrawOptionFeature(
            visualization_type=VisualizationType.histogram,
            title_x="Długość lewego kontekstu",
            title_y="Liczba wystąpień",
            other_label_name="-",
            map_key_to_label_name={},
        )


class LeftContextLengthFeatureChecker(FeatureChecker):
    name: str = FEATURE_NAME
    min_left_context_length: int
    max_left_context_length: int

    def _is_accepted_value(self, feature: Union[BaseFeature, LeftContextLengthFeature], _: FeatureStatistics) -> bool:
        feature = self.check_expected_type(LeftContextLengthFeature, feature)
        return self.min_left_context_length <= feature.left_context_length <= self.max_left_context_length

    @staticmethod
    def get_checker_name() -> str:
        return FEATURE_NAME
