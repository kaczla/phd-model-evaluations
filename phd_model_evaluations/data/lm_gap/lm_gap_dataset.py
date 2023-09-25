from typing import List

from pydantic import BaseModel

from phd_model_evaluations.data.lm_gap.lm_gap_dataset_data import LMGapDatasetData
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration


class LMGapDataset(BaseModel):
    train: List[LMGapDatasetData]
    valid: List[LMGapDatasetData]
    test: List[LMGapDatasetData]
    metric: MetricConfiguration
