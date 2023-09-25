from typing import List

from pydantic import BaseModel

from phd_model_evaluations.data.loaded_dataset_data import LoadedDatasetData
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration


class LoadedDataset(BaseModel):
    train: List[LoadedDatasetData]
    valid: List[LoadedDatasetData]
    test: List[LoadedDatasetData]
    metric: MetricConfiguration

    def get_total_train_lines(self) -> int:
        return sum(len(dataset_data.data_text) for dataset_data in self.train)

    def get_total_valid_lines(self) -> int:
        return sum(len(dataset_data.data_text) for dataset_data in self.valid)

    def get_total_test_lines(self) -> int:
        return sum(len(dataset_data.data_text) for dataset_data in self.test)
