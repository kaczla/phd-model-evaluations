from typing import List

from phd_model_evaluations.data.dataset_info import DatasetInfo
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration
from phd_model_evaluations.utils.type_utils import TYPE_DATASET_LABELS


class DatasetConfiguration(DatasetInfo):
    input_labels: List[str]
    target_label: str
    target_label_values: TYPE_DATASET_LABELS
    metric_configuration: MetricConfiguration
