#!/usr/bin/env python3

import logging
from typing import List, Optional, Tuple

from transformers import AutoConfig, HfArgumentParser, PretrainedConfig, Trainer, TrainingArguments

from phd_model_evaluations.cli.evaluation.model.evaluate_model_arguments import DataEvaluationArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.data.results.evaluation_result import EvaluationResult
from phd_model_evaluations.train.data_utils import get_sequence_length_and_padding_strategy
from phd_model_evaluations.train.finetuning.classification_utils import (
    get_classification_label_to_id,
    get_classification_metrics_fn,
    get_classification_model,
    get_test_classification_data,
    preprocess_classification_data,
)
from phd_model_evaluations.train.trainer.trainer_utils import save_evaluation_statistics
from phd_model_evaluations.utils.dataset_utils import get_metric_for_best_model
from phd_model_evaluations.utils.model_utils import check_padding_token, get_best_model_path, get_tokenizer

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[ModelArguments, DataEvaluationArguments, LoggerArguments, TrainingArguments]:
    parser = HfArgumentParser(
        (ModelArguments, DataEvaluationArguments, LoggerArguments, TrainingArguments),
        description="Run evaluation on given data - it's support classification and regression.",
    )
    model_args, dataset_args, logger_args, training_args = parser.parse_args_into_dataclasses(
        args=cmd_args, look_for_args_file=False
    )
    return model_args, dataset_args, logger_args, training_args


def run_evaluation() -> None:
    model_args, dataset_args, logger_args, training_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    test_dataset_list, dataset_configuration = get_test_classification_data(dataset_args)

    model_args.model_name = (
        get_best_model_path(model_args.model_name) if dataset_args.find_best_model else model_args.model_name
    )
    LOGGER.info(f"Loading model from: {model_args.model_name}")
    config: PretrainedConfig = AutoConfig.from_pretrained(model_args.model_name)
    is_regression = config.num_labels in {0, 1}
    label_to_id = get_classification_label_to_id(config, dataset_configuration.target_label_values, is_regression)
    metric_for_best_model, _ = get_metric_for_best_model(
        dataset_configuration.dataset_name,
        dataset_configuration.configuration_name,
        training_args.metric_for_best_model,
        training_args.greater_is_better,
    )

    tokenizer = get_tokenizer(model_args)
    sequence_length, padding_strategy, data_collator = get_sequence_length_and_padding_strategy(
        model_args.model_name, tokenizer, model_args.sequence_length, model_args.pad_to_max_length
    )

    model = get_classification_model(model_args, config)
    check_padding_token(tokenizer, model)

    compute_metrics_fn = get_classification_metrics_fn(dataset_configuration.metric_configuration, is_regression)

    evaluation_outputs = []
    for test_dataset_data in test_dataset_list:
        LOGGER.info(f"Evaluating {test_dataset_data.dataset_info.set_name} set ...")
        test_dataset = preprocess_classification_data(
            test_dataset_data.dataset,
            tokenizer,
            sequence_length,
            padding_strategy,
            label_to_id,
            is_regression,
            dataset_configuration.input_labels,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics_fn,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        evaluation_metrics = trainer.evaluate()
        LOGGER.info(f"Evaluation metrics: {evaluation_metrics}")
        evaluation_outputs.append(
            EvaluationResult(
                dataset_name=dataset_args.dataset_name,
                set_name=test_dataset_data.dataset_info.set_name,
                test_file=str(dataset_args.test_file.absolute()),
                model_name=str(model_args.model_name),
                model_human_name=model_args.model_human_name,
                best_metric=evaluation_metrics[f"eval_{metric_for_best_model}"],
                metrics=evaluation_metrics,
            )
        )

    save_evaluation_statistics(training_args.output_dir, evaluation_outputs)
    LOGGER.info("Evaluation finished")


if __name__ == "__main__":
    run_evaluation()
