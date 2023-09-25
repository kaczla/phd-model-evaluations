from typing import Dict, Union

from phd_model_evaluations.data.feature.checker.base_feature import BaseFeature, FeatureChecker
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType

FEATURE_NAME = "right_context_length"


class RightContextLengthFeature(BaseFeature):
    name: str = FEATURE_NAME
    right_context_length: int

    def get_aggregation_data(self) -> Dict[int, int]:
        return {self.right_context_length: 1}

    @staticmethod
    def get_draw_option() -> DrawOptionFeature:
        return DrawOptionFeature(
            visualization_type=VisualizationType.histogram,
            title_x="Długość prawego kontekstu",
            title_y="Liczba wystąpień",
            other_label_name="-",
            map_key_to_label_name={},
        )


class RightContextLengthFeatureChecker(FeatureChecker):
    name: str = FEATURE_NAME
    min_right_context_length: int
    max_right_context_length: int

    def _is_accepted_value(self, feature: Union[BaseFeature, RightContextLengthFeature], _: FeatureStatistics) -> bool:
        feature = self.check_expected_type(RightContextLengthFeature, feature)
        return self.min_right_context_length <= feature.right_context_length <= self.max_right_context_length

    @staticmethod
    def get_checker_name() -> str:
        return FEATURE_NAME
