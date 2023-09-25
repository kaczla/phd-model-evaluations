from datasets import Dataset
from pydantic import BaseModel

from phd_model_evaluations.data.dataset_info import DatasetInfo
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration


class LoadedDatasetsMappingSingle(BaseModel):
    dataset_info: DatasetInfo
    dataset: Dataset
    metric: MetricConfiguration

    class Config:
        arbitrary_types_allowed = True
