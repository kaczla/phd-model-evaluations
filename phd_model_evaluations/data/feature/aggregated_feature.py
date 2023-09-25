from typing import Dict, Union

from pydantic import BaseModel

from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.statistics.score_statistics import ScoreStatistics
from phd_model_evaluations.utils.common_utils import get_key_from_directory_items


class AggregatedFeature(BaseModel):
    name: str
    data: Union[Dict[int, int], Dict[str, int]]
    total: int
    statistics: ScoreStatistics
    draw_option: DrawOptionFeature

    def sort_data(self) -> None:
        # It's not an elegant way to sort data, but otherwise `mypy` will raise error
        raw_data = [(key, value) for key, value in self.data.items()]
        raw_data.sort(key=get_key_from_directory_items)
        if not raw_data:
            return
        elif isinstance(raw_data[0][0], int):
            self.data = {int(key): value for key, value in raw_data}
        else:
            self.data = {str(key): value for key, value in raw_data}
