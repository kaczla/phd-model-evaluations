import json
import logging
from pathlib import Path
from typing import List

from phd_model_evaluations.data.statistics.loss_statistics import LossStatistics
from phd_model_evaluations.utils.common_utils import save_json_data, sort_json_data

LOGGER = logging.getLogger(__name__)


def load_loss_statistics(save_path: Path) -> List[LossStatistics]:
    if save_path.exists():
        LOGGER.info("Found loss statistics, adding into existing statistics")
        return [LossStatistics(**loss_statistics_dict) for loss_statistics_dict in json.loads(save_path.read_text())]

    return []


def save_loss_statistics(save_path: Path, loss_statistics: LossStatistics) -> None:
    all_loss_statistics = load_loss_statistics(save_path)
    all_loss_statistics.append(loss_statistics)

    all_loss_statistics_dict = sort_json_data(
        [loss_statistics.dict() for loss_statistics in all_loss_statistics], "model_name"
    )
    save_json_data(all_loss_statistics_dict, save_path)
