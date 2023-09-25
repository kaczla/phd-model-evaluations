from typing import List

from pydantic import BaseModel

from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.feature.line_features import LineFeatures


class FeatureData(BaseModel):
    features: List[LineFeatures]
    statistics: FeatureStatistics
