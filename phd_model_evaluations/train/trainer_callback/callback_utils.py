import logging
from typing import List

from transformers import TrainerCallback

from phd_model_evaluations.train.trainer_callback.early_stopping_with_logger import EarlyStoppingWithLoggerCallback
from phd_model_evaluations.train.trainer_callback.save_on_end_epoch import SaveOnEndEpochTrainerCallback
from phd_model_evaluations.train.trainer_callback.stop_at_empty_evaluation import StopAtEmptyEvaluation

LOGGER = logging.getLogger(__name__)


def get_callbacks(early_stopping_patience: int, early_stopping_threshold: float) -> List[TrainerCallback]:
    callbacks = [SaveOnEndEpochTrainerCallback(), StopAtEmptyEvaluation()]

    if early_stopping_patience > 0:
        LOGGER.info(
            f"Using early stopping with patience: {early_stopping_patience} and threshold: {early_stopping_threshold}"
        )
        callbacks.append(
            EarlyStoppingWithLoggerCallback(
                early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold
            )
        )

    return callbacks
