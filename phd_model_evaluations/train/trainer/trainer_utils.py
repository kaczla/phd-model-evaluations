import json
import logging
from pathlib import Path
from typing import List, Optional

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import TrainOutput

from phd_model_evaluations.data.results.evaluation_result import EvaluationResult

LOGGER = logging.getLogger(__name__)


def set_save_and_evaluate_arguments(
    saving_and_evaluation_ratio: Optional[float], epoch_examples: int, training_arguments: TrainingArguments
) -> None:
    if saving_and_evaluation_ratio is None:
        return

    number_steps = int(saving_and_evaluation_ratio * epoch_examples)
    LOGGER.info(
        f'Set saving and evaluation arguments to "step" strategy with steps: {number_steps}'
        f" (ratio {saving_and_evaluation_ratio})"
    )
    training_arguments.save_strategy = "steps"
    training_arguments.save_steps = number_steps
    training_arguments.evaluation_strategy = "steps"
    training_arguments.eval_steps = number_steps


def set_logging_arguments(
    logging_ratio: Optional[float], epoch_examples: int, training_arguments: TrainingArguments
) -> None:
    if logging_ratio is None:
        return

    number_steps = int(logging_ratio * epoch_examples)
    LOGGER.info(f'Set logging arguments to "step" strategy with steps: {number_steps} (ratio {logging_ratio})')
    training_arguments.logging_strategy = "steps"
    training_arguments.logging_steps = number_steps


def save_train_statistics_at_end_training(output_dir: str, trainer: Trainer, train_output: TrainOutput) -> None:
    LOGGER.info(f"Best score: {trainer.state.best_metric}")
    LOGGER.info(f"Best checkpoint path: {trainer.state.best_model_checkpoint}")
    trainer.save_state()
    output_path = Path(output_dir) / "train_output.json"
    output_path.write_text(json.dumps(train_output._asdict(), ensure_ascii=False, indent=4))
    LOGGER.info(f"Training statistics saved in: {output_path}")


def save_evaluation_statistics(output_dir: str, evaluation_outputs: List[EvaluationResult]) -> None:
    output_path = Path(output_dir) / "evaluation_output.json"

    evaluation_data = []
    if output_path.exists():
        LOGGER.debug("Adding evaluation statistics into existing statistics")
        evaluation_data = json.loads(output_path.read_text())

    evaluation_data.extend([evaluation_output.dict() for evaluation_output in evaluation_outputs])
    output_path.write_text(json.dumps(evaluation_data, ensure_ascii=False, indent=4))
    LOGGER.info(f"Evaluation saved in: {output_path}")
