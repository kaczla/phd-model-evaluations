import logging

from transformers import TrainingArguments

from phd_model_evaluations.data.arguments.model_arguments import ModelArguments

LOGGER = logging.getLogger(__name__)


def set_batch_size(model_arguments: ModelArguments, training_arguments: TrainingArguments) -> None:
    if (
        model_arguments.max_batch_size_with_gradient_accumulation is not None
        and model_arguments.max_batch_size_with_gradient_accumulation > 0
    ):
        max_batch_size_with_gradient_accumulation = model_arguments.max_batch_size_with_gradient_accumulation
        batch_size = training_arguments.per_device_train_batch_size
        if max_batch_size_with_gradient_accumulation % batch_size:
            raise RuntimeError(
                f"Cannot set gradient accumulation for batch size: {batch_size}"
                f" and max batch size with gradient accumulation: {max_batch_size_with_gradient_accumulation}"
                f" cannot divide {max_batch_size_with_gradient_accumulation}/{batch_size}!"
            )
        else:
            gradient_accumulation_steps = int(max_batch_size_with_gradient_accumulation // batch_size)
            # Do not change if not needed
            if gradient_accumulation_steps == training_arguments.gradient_accumulation_steps:
                return

            training_arguments.gradient_accumulation_steps = gradient_accumulation_steps
            LOGGER.info(
                f"Setting gradient accumulation to: {gradient_accumulation_steps} where batch_size: {batch_size} and"
                f" final batch size: {batch_size * gradient_accumulation_steps}"
            )
            assert (
                batch_size * gradient_accumulation_steps == max_batch_size_with_gradient_accumulation
            ), "Invalid final batch size"
