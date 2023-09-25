import logging
from typing import Dict, List, Optional

from pydantic import BaseModel

from phd_model_evaluations.data.results.table_data import TableData
from phd_model_evaluations.utils.model_names import filter_model_names

LOGGER = logging.getLogger(__name__)


class AggregatedResults(BaseModel):
    model_list: List[str]
    # metrics maps: dataset name -> metric name
    metrics: Dict[str, str]
    # results maps: dataset name -> model name with optional score
    results: Dict[str, Dict[str, Optional[float]]]

    def get_table_data(self, return_empty: bool = False) -> TableData:
        data_set_names = self.get_data_set_names()

        row_names: List[str] = []
        row_data: List[Dict[str, str]] = []
        one_line_row_names: List[str] = []

        # Add data for each models
        split_data = self.results
        for model_name in self.model_list:
            model_data = {}
            for data_set_name in data_set_names:
                if data_set_name not in split_data or model_name not in split_data[data_set_name]:
                    continue

                model_value = split_data[data_set_name][model_name]
                if model_value is not None:
                    model_data[data_set_name] = f"{model_value:.2f}"

            if return_empty or model_data:
                row_names.append(model_name)
                row_data.append({})
                one_line_row_names.append(model_name)
                row_names.append(f"{model_name} data")
                row_data.append(model_data)

        return TableData(
            column_names=data_set_names,
            row_names=row_names,
            row_data=row_data,
            one_line_row_names=one_line_row_names,
            skip_row_name=True,
        )

    def get_data_set_names(self) -> List[str]:
        # Get data set names from metrics
        all_data_set_names = list(self.metrics.keys())
        all_added_names = set(all_data_set_names)

        # Add data set names from data
        for data_set_name in self.results.keys():
            if data_set_name not in all_added_names:
                all_data_set_names.append(data_set_name)
                all_added_names.add(data_set_name)

        # Make predefined data set names first
        data_set_names = []
        added_names = set()
        for predefined_name in ["loss", "LM-GAP"]:
            if predefined_name in all_added_names:
                data_set_names.append(predefined_name)
                added_names.add(predefined_name)

        # Add other data set names
        for data_set_name in all_data_set_names:
            if data_set_name not in added_names:
                data_set_names.append(data_set_name)

        return data_set_names

    def filter_model_names(self, return_encoder: bool, return_decoder: bool, return_encoder_decoder: bool) -> None:
        filtered_model_list = filter_model_names(
            self.model_list, return_encoder, return_decoder, return_encoder_decoder
        )
        assert len(filtered_model_list) > 0, "Filtered model names cannot be empty!"
        LOGGER.info(f"Model names after filtering: {filtered_model_list}")
        LOGGER.info(f"Filtered from {len(self.model_list)} models to {len(filtered_model_list)} models")
        for data_set_name, data_set_dict in self.results.items():
            new_data = {
                model_name: data_set_dict[model_name]
                for model_name in filtered_model_list
                if model_name in data_set_dict
            }
            self.results[data_set_name] = new_data
        self.model_list = filtered_model_list
        return None
