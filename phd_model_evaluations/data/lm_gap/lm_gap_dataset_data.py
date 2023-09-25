from typing import List

from phd_model_evaluations.data.dataset_info import DatasetInfo
from phd_model_evaluations.data.lm_gap.lm_gap_line import LMGapLine
from phd_model_evaluations.utils.type_utils import (
    TYPE_DATASET_ELEMENT_LIST_DICT,
    TYPE_DATASET_FEATURE_TO_DEFINITION_DICT,
)


class LMGapDatasetData(DatasetInfo):
    lm_gap_lines: List[LMGapLine]
    raw_data: TYPE_DATASET_ELEMENT_LIST_DICT
    feature_definitions: TYPE_DATASET_FEATURE_TO_DEFINITION_DICT
