#!/usr/bin/env python3

"""
Example run:
CUDA_VISIBLE_DEVICES='0' TRANSFORMERS_CACHE='.cache/transformers' \
python -m phd_model_evaluations.cli.train.finetuning.run_finetuning \
  --model_name roberta-base \
  --train_file glue-lm-gap/train/dataset_data.json.xz \
  --validation_file glue-lm-gap/dev-0/dataset_data.json.xz \
  --dataset_name cola \
  --output_dir out/roberta-base-cola
"""

import logging
from typing import List, Optional, Tuple

from transformers import AutoConfig, HfArgumentParser, PretrainedConfig, Trainer, TrainingArguments
from transformers.trainer_utils import TrainOutput, set_seed

from phd_model_evaluations.cli.train.finetuning.run_finetuning_arguments import DataTrainingArguments
from phd_model_evaluations.data.arguments.arguments_utils import set_batch_size
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.train.data_utils import get_sequence_length_and_padding_strategy
from phd_model_evaluations.train.finetuning.classification_utils import (
    get_classification_dataset,
    get_classification_label_to_id,
    get_classification_metrics_fn,
    get_classification_model,
    preprocess_classification_data,
)
from phd_model_evaluations.train.trainer.trainer_utils import (
    save_train_statistics_at_end_training,
    set_logging_arguments,
    set_save_and_evaluate_arguments,
)
from phd_model_evaluations.train.trainer_callback.callback_utils import get_callbacks
from phd_model_evaluations.utils.dataset_utils import get_metric_for_best_model
from phd_model_evaluations.utils.model_utils import (
    check_padding_token,
    get_last_checkpoint_directory,
    get_tokenizer,
    update_config_model,
)

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[ModelArguments, DataTrainingArguments, LoggerArguments, TrainingArguments]:
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LoggerArguments, TrainingArguments),
        description="Run fine-tuning model on given data - it's support classification and regression.",
    )
    model_args, dataset_args, logger_args, training_args = parser.parse_args_into_dataclasses(
        args=cmd_args, look_for_args_file=False
    )

    if training_args.data_seed is None:
        training_args.data_seed = 43

    return model_args, dataset_args, logger_args, training_args


def run_fine_tuning() -> None:
    model_args, dataset_args, logger_args, training_args = parse_args()
    set_logging_from_logger_arguments(logger_args)
    set_batch_size(model_args, training_args)
    set_seed(training_args.seed)

    train_dataset, validation_dataset, dataset_configuration = get_classification_dataset(dataset_args)
    is_regression = len(dataset_configuration.target_label_values) == 0
    number_of_labels = 1 if is_regression else len(dataset_configuration.target_label_values)
    training_args.metric_for_best_model, training_args.greater_is_better = get_metric_for_best_model(
        dataset_configuration.dataset_name,
        dataset_configuration.configuration_name,
        training_args.metric_for_best_model,
        training_args.greater_is_better,
    )

    model_name = model_args.model_name

    config: PretrainedConfig = AutoConfig.from_pretrained(
        model_name, num_labels=number_of_labels, finetuning_task=dataset_args.dataset_name
    )
    config = update_config_model(model_args, config)
    label_to_id = get_classification_label_to_id(config, dataset_configuration.target_label_values, is_regression)

    tokenizer = get_tokenizer(model_args)
    sequence_length, padding_strategy, data_collator = get_sequence_length_and_padding_strategy(
        model_name,
        tokenizer,
        model_args.sequence_length,
        model_args.pad_to_max_length,
    )

    model = get_classification_model(model_args, config)
    check_padding_token(tokenizer, model)
    last_checkpoint = get_last_checkpoint_directory(training_args.output_dir)
    training_args.load_best_model_at_end = (
        training_args.load_best_model_at_end or model_args.early_stopping_patience > 0
    )

    train_dataset = preprocess_classification_data(
        train_dataset,
        tokenizer,
        sequence_length,
        padding_strategy,
        label_to_id,
        is_regression,
        dataset_configuration.input_labels,
        seed=training_args.data_seed,
    )
    validation_dataset = preprocess_classification_data(
        validation_dataset,
        tokenizer,
        sequence_length,
        padding_strategy,
        label_to_id,
        is_regression,
        dataset_configuration.input_labels,
    )
    compute_metrics_fn = get_classification_metrics_fn(dataset_configuration.metric_configuration, is_regression)
    train_iterations = len(train_dataset) // (
        training_args.train_batch_size * training_args.gradient_accumulation_steps
    )
    set_save_and_evaluate_arguments(dataset_args.saving_and_evaluation_ratio, train_iterations, training_args)
    set_logging_arguments(dataset_args.logging_ratio, train_iterations, training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=get_callbacks(model_args.early_stopping_patience, model_args.early_stopping_threshold),
    )

    train_output: TrainOutput = trainer.train(resume_from_checkpoint=last_checkpoint)
    save_train_statistics_at_end_training(training_args.output_dir, trainer, train_output)

    LOGGER.info("Fine-tuning finished")


if __name__ == "__main__":
    run_fine_tuning()
