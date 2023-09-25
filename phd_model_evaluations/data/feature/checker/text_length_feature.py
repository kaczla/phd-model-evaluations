from typing import Dict, Union

from phd_model_evaluations.data.feature.checker.base_feature import BaseFeature, FeatureChecker
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType

FEATURE_NAME = "text_length"


class TextLengthFeature(BaseFeature):
    name: str = FEATURE_NAME
    text_length: int

    def get_aggregation_data(self) -> Dict[int, int]:
        return {self.text_length: 1}

    @staticmethod
    def get_draw_option() -> DrawOptionFeature:
        return DrawOptionFeature(
            visualization_type=VisualizationType.histogram,
            title_x="Długość tekstu",
            title_y="Liczba wystąpień",
            other_label_name="-",
            map_key_to_label_name={},
        )


class TextLengthFeatureChecker(FeatureChecker):
    name: str = FEATURE_NAME
    min_text_length: int
    max_text_length: int

    def _is_accepted_value(self, feature: Union[BaseFeature, TextLengthFeature], _: FeatureStatistics) -> bool:
        feature = self.check_expected_type(TextLengthFeature, feature)
        return self.min_text_length <= feature.text_length <= self.max_text_length

    @staticmethod
    def get_checker_name() -> str:
        return FEATURE_NAME
