import logging
from typing import Any, Dict

import numpy as np
from transformers import EarlyStoppingCallback, TrainerControl, TrainerState, TrainingArguments

LOGGER = logging.getLogger(__name__)


class EarlyStoppingWithLoggerCallback(EarlyStoppingCallback):
    """This is a copy from transformers.trainer_callback.EarlyStoppingCallback with logging."""

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0) -> None:
        super().__init__(early_stopping_patience, early_stopping_threshold)

    def check_metric_value(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metric_value: float  # noqa: ARG002
    ) -> None:
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)  # type: ignore[operator]
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
            LOGGER.info(
                f"Early stopping patience increased:"
                f" {self.early_stopping_patience_counter} / {self.early_stopping_patience}"
            )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any],
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            LOGGER.warning(
                f"Early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            LOGGER.info("Early stopping patience reached - stopping training")
            control.should_training_stop = True
