from typing import Dict, Type, Union

from phd_model_evaluations.data.feature.checker.base_feature import BaseFeature, FeatureChecker
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType

FEATURE_NAME = "is_number"


class IsNumberFeature(BaseFeature):
    name: str = FEATURE_NAME
    is_number: bool

    def get_aggregation_data(self) -> Dict[str, int]:
        return {"is_number": int(self.is_number)}

    @staticmethod
    def get_draw_option() -> DrawOptionFeature:
        return DrawOptionFeature(
            visualization_type=VisualizationType.pie,
            title_x="-",
            title_y="-",
            other_label_name="Nie jest liczbą",
            map_key_to_label_name={"is_number": "Jest liczbą"},
        )

    @classmethod
    def from_gap(cls: Type["IsNumberFeature"], value: str) -> "IsNumberFeature":
        is_number = False
        try:
            float(value.replace(",", "").strip())
            is_number = True
        except ValueError:
            pass
        return IsNumberFeature(is_number=is_number)


class IsNumberFeatureChecker(FeatureChecker):
    name: str = FEATURE_NAME
    check_number: bool

    def _is_accepted_value(self, feature: Union[BaseFeature, IsNumberFeature], _: FeatureStatistics) -> bool:
        feature = self.check_expected_type(IsNumberFeature, feature)
        return bool(self.check_number and feature.is_number)

    @staticmethod
    def get_checker_name() -> str:
        return FEATURE_NAME
