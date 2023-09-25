from typing import List

from phd_model_evaluations.data.dataset_info import DatasetInfo
from phd_model_evaluations.data.loaded_dataset_text_line import LoadedDatasetTextLine
from phd_model_evaluations.utils.type_utils import TYPE_DATASET_FEATURE_TO_DEFINITION_DICT


class LoadedDatasetData(DatasetInfo):
    data_text: List[LoadedDatasetTextLine]
    feature_definitions: TYPE_DATASET_FEATURE_TO_DEFINITION_DICT

    def get_raw_data_keys(self) -> List[str]:
        if not self.data_text or not self.data_text[0].raw_data:
            return []

        return list(self.data_text[0].raw_data.keys())
