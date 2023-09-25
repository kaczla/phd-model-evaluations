import logging
from math import isnan
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import evaluate
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from transformers import (
    BatchEncoding,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.data.dataset_configuration import DatasetConfiguration
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration
from phd_model_evaluations.data.metric.metric_for_best_model import MetricForBestModel
from phd_model_evaluations.data.model.seq2seq_task_specific_paramaters import Seq2SeqTaskSpecificParameters
from phd_model_evaluations.data.model.task_specific_paramaters import TaskSpecificParameters
from phd_model_evaluations.utils.model_utils import get_sequence_length
from phd_model_evaluations.utils.type_utils import SEQ2SEQ_MODEL_TYPES

DATASET_NAMES_TO_PREFIX_AND_KEY_TEXT: Dict[Tuple[str, ...], List[Tuple[str, str]]] = {
    ("glue", "cola"): [
        ("cola sentence: ", "sentence"),
    ],
    ("glue", "sst2"): [
        ("sst2 sentence: ", "sentence"),
    ],
    ("glue", "mrpc"): [
        ("mrpc sentence1: ", "sentence1"),
        ("sentence2: ", "sentence2"),
    ],
    ("glue", "qqp"): [
        ("qqp question1: ", "question1"),
        ("question2: ", "question2"),
    ],
    ("glue", "stsb"): [
        ("stsb sentence1: ", "sentence1"),
        ("sentence2: ", "sentence2"),
    ],
    ("glue", "mnli"): [
        ("mnli hypothesis: ", "hypothesis"),
        ("premise: ", "premise"),
    ],
    ("glue", "qnli"): [
        ("qnli question: ", "question"),
        ("sentence: ", "sentence"),
    ],
    ("glue", "rte"): [
        ("rte sentence1: ", "sentence1"),
        ("sentence2: ", "sentence2"),
    ],
}

SEQ2SEQ_TARGET_LABEL = "label"
SEQ2SEQ_TARGET_LABELS = "labels"

LOGGER = logging.getLogger(__name__)


def get_target_max_length(train_dataset: Dataset, validation_dataset: Dataset) -> int:
    train_target_max_length: int = max((len(d) for d in train_dataset[SEQ2SEQ_TARGET_LABELS]))
    valid_target_max_length: int = max((len(d) for d in validation_dataset[SEQ2SEQ_TARGET_LABELS]))
    target_max_length = max(train_target_max_length, valid_target_max_length)
    assert target_max_length > 0, "Max target length cannot be empty!"
    return target_max_length


def update_generation_max_length(
    model_arguments: ModelArguments,
    training_arguments: Seq2SeqTrainingArguments,
    train_dataset: Dataset,
    validation_dataset: Dataset,
) -> None:
    if model_arguments.dynamic_generation_length:
        max_length = get_target_max_length(train_dataset, validation_dataset)
        LOGGER.info(
            f"Using generation max length base on dataset target/label data: {max_length}"
            f" (previous length: {training_arguments.generation_max_length})"
        )
        training_arguments.generation_max_length = max_length


def get_prefix_and_key_text(dataset_name: str, configuration_name: Optional[str] = None) -> List[Tuple[str, str]]:
    if configuration_name is None:
        return DATASET_NAMES_TO_PREFIX_AND_KEY_TEXT[(dataset_name,)]

    return DATASET_NAMES_TO_PREFIX_AND_KEY_TEXT[(dataset_name, configuration_name)]


def get_sequence_length_and_padding_strategy_seq2seq(
    model: SEQ2SEQ_MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    sequence_length: Optional[int],
    pad_to_max_length: bool,
    compute_loss: bool = False,
) -> Tuple[int, Union[str, bool], Optional[Callable[[List[Any]], Any]]]:
    sequence_length = get_sequence_length(sequence_length, tokenizer)
    padding_strategy: Union[str, bool]
    # Pad to the sequence length or to the maximum length in the batch
    if pad_to_max_length:
        padding_strategy, data_collator = "max_length", default_data_collator
    else:
        label_pad_token_id = -100 if compute_loss else tokenizer.pad_token_id
        padding_strategy = False
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=label_pad_token_id, return_tensors="pt"
        )

    return sequence_length, padding_strategy, data_collator


def preprocess_seq2seq_data(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    sequence_length: int,
    target_length: int,
    padding_strategy: Union[str, bool],
    is_regression: bool,
    id_to_label: Dict[int, str],
    prefix_and_key_text: List[Tuple[str, str]],
    dataset_name: str = "",
    seed: Optional[int] = None,
) -> Dataset:
    def preprocess_fn(examples: LazyBatch) -> BatchEncoding:
        result = tokenizer(
            *[[prefix + example for example in examples[key]] for prefix, key in prefix_and_key_text],
            padding=padding_strategy,
            max_length=sequence_length,
            truncation=True,
        )
        if is_regression:
            labels = tokenizer(
                [str(value) for value in examples[SEQ2SEQ_TARGET_LABEL]],
                padding=padding_strategy,
                max_length=target_length,
                truncation=True,
            )
        else:
            labels = tokenizer(
                [id_to_label[label_index] for label_index in examples[SEQ2SEQ_TARGET_LABEL]],
                padding=padding_strategy,
                max_length=target_length,
                truncation=True,
            )
        result[SEQ2SEQ_TARGET_LABELS] = labels["input_ids"]
        return result

    # Remove all not required columns after preprocessing
    columns_to_remove = list(
        {feature for feature in list(dataset.features.keys()) if feature not in {SEQ2SEQ_TARGET_LABELS}}
    )

    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        load_from_cache_file=False,
        remove_columns=columns_to_remove,
        desc=f"Running tokenization on {dataset_name} set" if dataset_name else "Running tokenization",
    )

    if seed is None:
        return dataset

    dataset = dataset.shuffle(seed=seed)
    return dataset


def parse_float_label(value: str, default_value: float) -> float:
    try:
        return float(value)
    except ValueError:
        return default_value


def get_seq2seq_metrics_fn(
    metric_configuration: MetricConfiguration,
    tokenizer: PreTrainedTokenizer,
    label_to_id: Dict[Union[str, int], int],
    is_regression: bool,
    metric_for_best_model: MetricForBestModel,
) -> Callable[[EvalPrediction], Dict[str, Any]]:
    LOGGER.info(f"Loading metric: {metric_configuration}")
    metric = evaluate.load(metric_configuration.name, config_name=metric_configuration.configuration_name)

    def compute_metrics(eval_prediction: EvalPrediction) -> Dict[str, Any]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = [prediction.strip() for prediction in decoded_predictions]
        decoded_labels = [label.strip() for label in decoded_labels]
        result: Dict[str, Any]
        if is_regression:
            # Return 0 value if cannot convert generated value
            decoded_predictions_values = [parse_float_label(prediction, 0.0) for prediction in decoded_predictions]
            decoded_labels_values = [parse_float_label(label, 0.0) for label in decoded_labels]
            result = metric.compute(predictions=decoded_predictions_values, references=decoded_labels_values)

            # Check is NaN
            if (
                metric_for_best_model.name in result
                and isinstance(result[metric_for_best_model.name], float)
                and isnan(result[metric_for_best_model.name])
            ):
                new_value = 0.0 if metric_for_best_model.greater_is_better else 1_000_000.0
                LOGGER.error(f"Detect NaN metric! Using value: {new_value}")
                result[metric_for_best_model.name] = new_value

        elif label_to_id:
            # Return label ID = 0 if cannot convert label to ID
            decoded_predictions_ids = [label_to_id.get(prediction, 0) for prediction in decoded_predictions]
            decoded_labels_ids = [label_to_id[label] for label in decoded_labels]
            result = metric.compute(predictions=decoded_predictions_ids, references=decoded_labels_ids)

        else:
            result = metric.compute(predictions=decoded_predictions, references=decoded_labels)

        return result

    return compute_metrics


def get_task_specific_parameters(
    model_arguments: ModelArguments,
    training_arguments: Seq2SeqTrainingArguments,
    dataset_configuration: DatasetConfiguration,
    label_to_id: Dict[Union[str, int], int],
    id_to_label: Dict[int, str],
    prefix_and_key_text: List[Tuple[str, str]],
) -> Seq2SeqTaskSpecificParameters:
    assert model_arguments.sequence_length is not None, "Sequence length cannot be empty!"
    return Seq2SeqTaskSpecificParameters(
        dataset_configuration=dataset_configuration,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        sequence_length=model_arguments.sequence_length,
        prefix_and_key_text=prefix_and_key_text,
        target_length=model_arguments.target_length,
        dynamic_generation_length=model_arguments.dynamic_generation_length,
        generation_max_length=training_arguments.generation_max_length,
        generation_num_beams=training_arguments.generation_num_beams,
    )


def get_task_specific_parameters_from_config(config: PretrainedConfig, task_name: str) -> Seq2SeqTaskSpecificParameters:
    return Seq2SeqTaskSpecificParameters(**config.task_specific_params[task_name])


def set_task_specific_parameters_in_configuration(
    config: PretrainedConfig, task_name: str, task_specific_parameters: TaskSpecificParameters
) -> None:
    if config.task_specific_params is None:
        config.task_specific_params = {}

    config.task_specific_params[task_name] = task_specific_parameters.dict()
