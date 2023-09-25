#!/usr/bin/env python3

import logging
from typing import List, Optional, Tuple

from transformers import (
    AutoConfig,
    HfArgumentParser,
    PretrainedConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from phd_model_evaluations.cli.evaluation.model.evaluate_model_arguments import DataEvaluationArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.data.metric.metric_for_best_model import MetricForBestModel
from phd_model_evaluations.data.results.evaluation_result import EvaluationResult
from phd_model_evaluations.train.finetuning.classification_utils import get_test_classification_data
from phd_model_evaluations.train.finetuning.seq2seq_utils import (
    get_seq2seq_metrics_fn,
    get_sequence_length_and_padding_strategy_seq2seq,
    get_task_specific_parameters_from_config,
    preprocess_seq2seq_data,
)
from phd_model_evaluations.train.trainer.trainer_utils import save_evaluation_statistics
from phd_model_evaluations.utils.dataset_utils import get_metric_for_best_model
from phd_model_evaluations.utils.model_utils import (
    check_padding_token,
    get_best_model_path,
    get_seq2seq_model,
    get_tokenizer,
)

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[ModelArguments, DataEvaluationArguments, LoggerArguments, Seq2SeqTrainingArguments]:
    parser = HfArgumentParser(
        (ModelArguments, DataEvaluationArguments, LoggerArguments, Seq2SeqTrainingArguments),
        description="Run seq2seq evaluation on given data - it's support classification and regression.",
    )
    model_args, dataset_args, logger_args, training_args = parser.parse_args_into_dataclasses(
        args=cmd_args, look_for_args_file=False
    )

    if training_args.data_seed is None:
        training_args.data_seed = 43

    training_args.predict_with_generate = True

    return model_args, dataset_args, logger_args, training_args


def run_seq2seq_evaluation() -> None:
    model_args, dataset_args, logger_args, training_args = parse_args()
    set_logging_from_logger_arguments(logger_args)
    set_seed(training_args.seed)

    test_dataset_list, dataset_configuration = get_test_classification_data(dataset_args)

    model_args.model_name = (
        get_best_model_path(model_args.model_name) if dataset_args.find_best_model else model_args.model_name
    )
    LOGGER.info(f"Loading model from: {model_args.model_name}")
    config: PretrainedConfig = AutoConfig.from_pretrained(model_args.model_name)
    task_specific_parameters = get_task_specific_parameters_from_config(config, dataset_configuration.get_name())
    is_regression = len(task_specific_parameters.id_to_label) == 0
    metric_for_best_model_name, greater_is_better = get_metric_for_best_model(
        dataset_configuration.dataset_name,
        dataset_configuration.configuration_name,
        training_args.metric_for_best_model,
        training_args.greater_is_better,
    )
    metric_for_best_model = MetricForBestModel(name=metric_for_best_model_name, greater_is_better=greater_is_better)

    tokenizer = get_tokenizer(model_args)
    model = get_seq2seq_model(model_args)
    check_padding_token(tokenizer, model)
    # Use created configuration for generating predictions - do not need to reload them
    model.generation_config._from_model_config = False

    sequence_length, padding_strategy, data_collator = get_sequence_length_and_padding_strategy_seq2seq(
        model, tokenizer, model_args.sequence_length, model_args.pad_to_max_length
    )
    LOGGER.info(f"Using sequence length for input: {sequence_length} and target: {model_args.target_length}")
    training_args.generation_max_length = task_specific_parameters.generation_max_length
    training_args.generation_num_beams = task_specific_parameters.generation_num_beams
    LOGGER.info(
        f"Using generation max length: {training_args.generation_max_length} and"
        f" beam search: {training_args.generation_num_beams}"
    )

    compute_metrics_fn = get_seq2seq_metrics_fn(
        dataset_configuration.metric_configuration,
        tokenizer,
        task_specific_parameters.label_to_id,
        is_regression,
        metric_for_best_model,
    )

    evaluation_outputs = []
    for test_dataset_data in test_dataset_list:
        LOGGER.info(f"Evaluating {test_dataset_data.dataset_info.set_name} set ...")
        test_dataset = preprocess_seq2seq_data(
            test_dataset_data.dataset,
            tokenizer,
            sequence_length,
            model_args.target_length,
            padding_strategy,
            is_regression,
            task_specific_parameters.id_to_label,
            task_specific_parameters.prefix_and_key_text,
        )
        trainer = Seq2SeqTrainer(
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
                best_metric=evaluation_metrics[f"eval_{metric_for_best_model.name}"],
                metrics=evaluation_metrics,
            )
        )

    save_evaluation_statistics(training_args.output_dir, evaluation_outputs)
    LOGGER.info("Evaluation finished")


if __name__ == "__main__":
    run_seq2seq_evaluation()
