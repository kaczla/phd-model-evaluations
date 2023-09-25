from typing import Dict, Union

from phd_model_evaluations.data.feature.checker.base_feature import BaseFeature, FeatureChecker
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType

FEATURE_NAME = "gap_with_punctuation"


class GapWithPunctuationFeature(BaseFeature):
    name: str = FEATURE_NAME
    start_with_punctuation: bool
    end_with_punctuation: bool
    has_punctuation: bool
    is_punctuation: bool

    def get_aggregation_data(self) -> Dict[str, int]:
        data = {
            "start_with_punctuation": 0,
            "end_with_punctuation": 0,
            "start_and_end_with_punctuation": 0,
            "has_punctuation": 0,
            "is_punctuation": 0,
        }
        if self.is_punctuation:
            data["is_punctuation"] = 1
        elif self.start_with_punctuation and self.end_with_punctuation:
            data["start_and_end_with_punctuation"] = 1
        elif self.start_with_punctuation:
            data["start_with_punctuation"] = 1
        elif self.end_with_punctuation:
            data["end_with_punctuation"] = 1
        elif self.has_punctuation:
            data["has_punctuation"] = 1
        return data

    @staticmethod
    def get_draw_option() -> DrawOptionFeature:
        return DrawOptionFeature(
            visualization_type=VisualizationType.pie,
            title_x="-",
            title_y="-",
            other_label_name="Nie posiada znaków interpunkcyjnych",
            map_key_to_label_name={
                "start_with_punctuation": "Zaczyna się znakiem interpunkcyjnych",
                "end_with_punctuation": "Kończy się znakiem interpunkcyjnym",
                "start_and_end_with_punctuation": "Zaczyna i kończy się znakiem interpunkcyjnym",
                "has_punctuation": "Posiada znak interpunkcyjny",
                "is_punctuation": "Składa się ze znaków interpunkcyjnych",
            },
        )


class GapWithPunctuationFeatureChecker(FeatureChecker):
    name: str = FEATURE_NAME
    check_start_with_punctuation: bool
    check_end_with_punctuation: bool
    check_has_punctuation: bool
    check_is_punctuation: bool

    def _is_accepted_value(self, feature: Union[BaseFeature, GapWithPunctuationFeature], _: FeatureStatistics) -> bool:
        feature = self.check_expected_type(GapWithPunctuationFeature, feature)
        return bool(
            (self.check_start_with_punctuation and feature.start_with_punctuation)
            or (self.check_end_with_punctuation and feature.end_with_punctuation)
            or (self.check_has_punctuation and feature.has_punctuation)
            or (self.check_is_punctuation and feature.is_punctuation)
        )

    @staticmethod
    def get_checker_name() -> str:
        return FEATURE_NAME
