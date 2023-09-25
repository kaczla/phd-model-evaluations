from pydantic import BaseModel

from phd_model_evaluations.utils.type_utils import TYPE_DATASET_DICT


class LoadedDatasetTextLine(BaseModel):
    text: str
    raw_data: TYPE_DATASET_DICT
