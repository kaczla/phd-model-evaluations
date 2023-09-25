from collections import defaultdict
from typing import Dict, List

from pydantic import BaseModel

from phd_model_evaluations.data.loaded_dataset_mapping_single import LoadedDatasetsMappingSingle


class LoadedDatasetsMapping(BaseModel):
    dataset_name_to_dataset_data: Dict[str, List[LoadedDatasetsMappingSingle]] = defaultdict(list)

    def total_dataset_data(self) -> int:
        return sum(len(dataset_data) for dataset_data in self.dataset_name_to_dataset_data.values())

    def total_dataset(self) -> int:
        return len(self.dataset_name_to_dataset_data)
