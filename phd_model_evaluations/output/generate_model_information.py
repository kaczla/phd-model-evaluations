import logging
from typing import List

from phd_model_evaluations.data.model.model_info import ModelInfo
from phd_model_evaluations.data.results.table_data import TableData
from phd_model_evaluations.utils.model_names import MAP_MODEL_NAME_TO_HUMAN_NAME

LOGGER = logging.getLogger(__name__)


def get_model_information(model_name: str) -> ModelInfo:
    LOGGER.info(f"Generating model information for: {model_name}")
    return ModelInfo(name=model_name, human_name=MAP_MODEL_NAME_TO_HUMAN_NAME.get(model_name, ""))


def generate_model_information(model_names: List[str]) -> List[ModelInfo]:
    return [get_model_information(model_name) for model_name in model_names]


def convert_model_information_to_table_data(model_parameters_list: List[ModelInfo]) -> TableData:
    model_names = []
    row_data = []
    for model_parameters in model_parameters_list:
        model_names.append(model_parameters.human_name)
        row_data.append(model_parameters.get_table_row())

    return TableData(
        column_names=ModelInfo.get_table_header(),
        row_names=model_names,
        row_data=row_data,
        one_line_row_names=[],
        skip_row_name=True,
    )
