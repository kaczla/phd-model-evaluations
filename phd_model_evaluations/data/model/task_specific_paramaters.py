from typing import Dict, Union

from pydantic import BaseModel

from phd_model_evaluations.data.dataset_configuration import DatasetConfiguration


class TaskSpecificParameters(BaseModel):
    dataset_configuration: DatasetConfiguration
    label_to_id: Dict[Union[str, int], int]
    id_to_label: Dict[int, str]
    sequence_length: int
