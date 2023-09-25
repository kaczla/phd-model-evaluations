import logging
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

LOGGER = logging.getLogger(__name__)


class StopAtEmptyEvaluation(TrainerCallback):
    """Callback for stopping training if found empty/null evaluation."""

    def __init__(self, max_empty_metrics: int = 3) -> None:
        super().__init__()
        self.max_empty_metrics = max_empty_metrics
        self.counter_empty_metrics = 0

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any  # noqa: ARG002
    ) -> None:
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = kwargs["metrics"].get(metric_to_check)

        if metric_value is None or metric_value == 0.0:
            self.counter_empty_metrics += 1
            if self.counter_empty_metrics >= self.max_empty_metrics:
                control.should_training_stop = True
                LOGGER.error(f"Stop training, empty metric value: {metric_value}")
            else:
                LOGGER.error(
                    f"Detect empty metric value: {metric_value}"
                    f" - reach {self.counter_empty_metrics} / {self.max_empty_metrics} tries"
                )
        else:
            self.counter_empty_metrics = 0
