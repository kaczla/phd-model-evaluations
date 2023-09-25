from abc import ABC, abstractmethod
from itertools import chain
from typing import Dict, Generic, List, Type, TypeVar

import numpy as np
from pydantic import BaseModel

from phd_model_evaluations.data.feature.aggregated_feature import AggregatedFeature
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.statistics.score_statistics import ScoreStatistics
from phd_model_evaluations.utils.common_utils import merge_dictionaries

BaseFeatureType = TypeVar("BaseFeatureType", bound="BaseFeature")
T = TypeVar("T")


class BaseFeature(BaseModel, Generic[T], ABC):
    name: str

    @abstractmethod
    def get_aggregation_data(self) -> Dict[T, int]:
        pass

    @staticmethod
    @abstractmethod
    def get_draw_option() -> DrawOptionFeature:
        pass

    @staticmethod
    def get_min_frequency() -> int:
        return 1

    @staticmethod
    def get_statistics(data: Dict[T, int]) -> ScoreStatistics:
        # Generate statistics for all float/int keys
        if all((isinstance(key, (int, float)) for key in data.keys())):
            array = np.array(list(chain.from_iterable(([k] * v for k, v in data.items()))))
            return ScoreStatistics(
                total_elements=array.size,
                max_value=array.max(),
                min_value=array.min(),
                avg_value=array.mean(),
                std_value=array.std(),
            )

        return ScoreStatistics(total_elements=0, max_value=0.0, min_value=0.0, avg_value=0.0, std_value=0.0)

    @classmethod
    def aggregate(
        cls: Type["BaseFeature"], features: List["BaseFeature"], sort_data: bool = False
    ) -> AggregatedFeature:
        feature = features[0]
        data = merge_dictionaries(
            [feature.get_aggregation_data() for feature in features], min_frequency_items=cls.get_min_frequency()
        )
        aggregation = AggregatedFeature(
            name=feature.name,
            data=data,
            total=len(features),
            statistics=cls.get_statistics(data),
            draw_option=cls.get_draw_option(),
        )

        if sort_data:
            aggregation.sort_data()

        return aggregation


class FeatureChecker(BaseModel, ABC):
    name: str
    invert_checking: bool

    @abstractmethod
    def _is_accepted_value(self, feature: BaseFeature, statistics: FeatureStatistics) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def get_checker_name() -> str:
        pass

    def is_accepted_value(self, feature: BaseFeature, statistics: FeatureStatistics) -> bool:
        return self.get_checking_result(self._is_accepted_value(feature, statistics))

    def get_checking_result(self, result: bool) -> bool:
        if self.invert_checking:
            return not result

        return result

    @staticmethod
    def check_expected_type(expected_type: Type[BaseFeatureType], object_to_check: BaseFeature) -> BaseFeatureType:
        if not isinstance(object_to_check, expected_type):
            raise RuntimeError(f"Expected type: {expected_type.name} but got: {object_to_check.__class__.__name__}")

        return object_to_check
