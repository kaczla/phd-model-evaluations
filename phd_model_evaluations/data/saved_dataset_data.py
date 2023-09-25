from pydantic import BaseModel

from phd_model_evaluations.data.dataset_info import DatasetInfo
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration
from phd_model_evaluations.utils.type_utils import TYPE_DATASET_DICT, TYPE_DATASET_FEATURE_TO_DEFINITION_DICT


class SavedDatasetsData(BaseModel):
    dataset_name: DatasetInfo
    metric: MetricConfiguration
    feature_definitions: TYPE_DATASET_FEATURE_TO_DEFINITION_DICT
    data: TYPE_DATASET_DICT
