import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import evaluate
import numpy as np
from datasets import ClassLabel, Dataset
from datasets.arrow_dataset import _concatenate_map_style_datasets
from datasets.formatting.formatting import LazyBatch
from transformers import (
    AutoModelForSequenceClassification,
    BatchEncoding,
    BertForSequenceClassification,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedTokenizer,
    RobertaForSequenceClassification,
)

from phd_model_evaluations.cli.evaluation.model.evaluate_model_arguments import DataEvaluationArguments
from phd_model_evaluations.cli.train.finetuning.run_finetuning_arguments import DataTrainingArguments
from phd_model_evaluations.data.arguments.model_arguments import ModelArguments
from phd_model_evaluations.data.dataset_configuration import DatasetConfiguration
from phd_model_evaluations.data.loaded_dataset_mapping_single import LoadedDatasetsMappingSingle
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration
from phd_model_evaluations.data.model.architecture_type import ArchitectureType
from phd_model_evaluations.model.custom_model_for_sequence_classification import CustomModelForSequenceClassification
from phd_model_evaluations.utils.dataset_utils import (
    get_dataset_feature_list,
    load_dataset_dict_from_raw_data_file,
    trim_dataset,
)
from phd_model_evaluations.utils.model_utils import get_auto_model, get_model_architecture
from phd_model_evaluations.utils.type_utils import TYPE_DATASET_LABELS

LOGGER = logging.getLogger(__name__)


ACCEPTED_CLASSIFICATION_INPUT_FEATURES = [
    ["sentence"],
    ["sentence1", "sentence2"],
    ["premise", "hypothesis"],
    ["question", "sentence"],
    ["question1", "question2"],
]

TARGET_LABEL = "label"

CLASSIFICATION_MODEL_TYPES = Union[
    RobertaForSequenceClassification, BertForSequenceClassification, CustomModelForSequenceClassification
]


def get_dataset_text_keys(dataset_name: str, dataset: Dataset) -> List[str]:
    features = get_dataset_feature_list(dataset, ACCEPTED_CLASSIFICATION_INPUT_FEATURES)
    LOGGER.debug(f"Using features {features} as text features in dataset: {dataset_name}")
    if len(features) <= 1:
        return [features[0]]
    elif len(features) > 2:
        LOGGER.warning(f"Using first 2 features from {features} as text features in dataset: {dataset_name}")

    return [features[0], features[1]]


def check_regression(dataset_name: str, uniq_values: List[Any]) -> None:
    LOGGER.debug(f"Detect regression in dataset: {dataset_name}")

    if len(uniq_values) <= 10:
        LOGGER.error(f"Found low number of uniq values ({len(uniq_values)}) for regression in dataset: {dataset_name}")


def check_number_of_class(dataset_name: str, number_of_labels: int) -> None:
    LOGGER.debug(f"Detect classification with {number_of_labels} classes in dataset: {dataset_name}")

    if number_of_labels <= 1:
        LOGGER.error(f"Found invalid number of classes ({number_of_labels} classes) in dataset: {dataset_name}")
    elif number_of_labels > 10:
        LOGGER.warning(f"Found more than 10 classes ({number_of_labels} classes) in dataset: {dataset_name}")


def get_labels_from_dataset(dataset_name: str, dataset: Dataset) -> TYPE_DATASET_LABELS:
    if TARGET_LABEL not in dataset.features:
        raise RuntimeError(f'Cannot find feature name "{TARGET_LABEL}" for classification for dataset: {dataset_name}')
    # Get number of classes if are "string" labels
    if isinstance(dataset.features[TARGET_LABEL], ClassLabel):
        check_number_of_class(dataset_name, len(dataset.features[TARGET_LABEL].names))
        names: TYPE_DATASET_LABELS = dataset.features[TARGET_LABEL].names
        return names
    # Detect regression - labels are float not integer type
    elif dataset.features[TARGET_LABEL].dtype in {"float32", "float64"}:
        check_regression(dataset_name, dataset.unique(TARGET_LABEL))
        # Weird mypy behavior
        empty_list: List[str] = []
        return empty_list

    labels: TYPE_DATASET_LABELS = dataset.unique(TARGET_LABEL)
    labels.sort()
    check_number_of_class(dataset_name, len(labels))
    return labels


def get_classification_model(model_arguments: ModelArguments, config: PretrainedConfig) -> CLASSIFICATION_MODEL_TYPES:
    architectures = config.to_dict().get("architectures", [])
    if architectures and "CustomModelForSequenceClassification" in architectures:
        LOGGER.info(f"Loading classification model: {CustomModelForSequenceClassification.__name__}")
        # Import used class as base model
        model_cls = getattr(__import__("transformers"), config.base_model_class)
        classifier_model = CustomModelForSequenceClassification.from_pretrained(
            model_arguments.model_name, config=config, ignore_mismatched_sizes=False, model_cls=model_cls
        )
        LOGGER.info(f"Loaded classification model: {CustomModelForSequenceClassification.__name__}")
        return classifier_model

    if model_arguments.use_custom_model:
        LOGGER.info(f"Using custom models: {CustomModelForSequenceClassification.__class__.__name__}")
        architecture_type = get_model_architecture(model_arguments)
        if architecture_type in {ArchitectureType.encoder, ArchitectureType.decoder}:
            config.classifier_dropout = 0.1
            config.has_cls_token = architecture_type == ArchitectureType.encoder
            model = get_auto_model(model_arguments)
            config.base_model_class = model.__class__.__name__
        else:
            # 7520661157024794
            RuntimeError(f"Not supported custom model for architecture type: {architecture_type}")

        LOGGER.info(f"Loading classification model: {CustomModelForSequenceClassification.__name__}")
        classifier_model = CustomModelForSequenceClassification(config, model.__class__, model=model)
        LOGGER.info(f"Loaded classification model: {CustomModelForSequenceClassification.__name__}")
        return classifier_model

    return AutoModelForSequenceClassification.from_pretrained(
        model_arguments.model_name, config=config, ignore_mismatched_sizes=False
    )


def check_dataset_keys(
    train_text_key_names: List[str],
    valid_text_key_names: List[str],
    train_name: str = "train",
    valid_name: str = "validation",
) -> None:
    if train_text_key_names != valid_text_key_names:
        raise RuntimeError(
            f"Detect invalid features name for text, for {train_name} set found: {train_text_key_names}"
            f" and for {valid_name} set found: {valid_text_key_names}"
        )


def check_dataset_labels(
    train_labels: TYPE_DATASET_LABELS,
    valid_labels: TYPE_DATASET_LABELS,
    train_name: str = "train",
    valid_name: str = "validation",
) -> None:
    if train_labels != valid_labels:
        raise RuntimeError(
            f"Detect mismatched labels, for {train_name} set found: {train_labels}"
            f" and for {valid_name} set found: {valid_labels}"
        )


def check_dataset_metrics(train_metrics: MetricConfiguration, valid_metrics: MetricConfiguration) -> None:
    if train_metrics != valid_metrics:
        raise RuntimeError(
            f"Detect mismatched metrics, for train found: {train_metrics} and for validation found: {valid_metrics}"
        )


def check_datasets_keys_and_labels(dataset_list: List[LoadedDatasetsMappingSingle], set_name: str) -> None:
    # Make sure each of datasets have the same keys and labels
    for dataset_index in range(len(dataset_list) - 1):
        dataset_1 = dataset_list[dataset_index]
        dataset_name_1 = f"{set_name}-{dataset_1.dataset_info.set_name}"
        dataset_2 = dataset_list[dataset_index + 1]
        dataset_name_2 = f"{set_name}-{dataset_2.dataset_info.set_name}"

        dataset_1_text_key_names = get_dataset_text_keys(dataset_name_1, dataset_1.dataset)
        dataset_2_text_key_names = get_dataset_text_keys(dataset_name_2, dataset_2.dataset)
        check_dataset_keys(
            dataset_1_text_key_names,
            dataset_2_text_key_names,
            train_name=dataset_name_1,
            valid_name=dataset_name_2,
        )

        dataset_1_dataset_labels = get_labels_from_dataset(dataset_name_1, dataset_1.dataset)
        dataset_2_dataset_labels = get_labels_from_dataset(dataset_name_2, dataset_2.dataset)
        check_dataset_labels(
            dataset_1_dataset_labels,
            dataset_2_dataset_labels,
            train_name=dataset_name_1,
            valid_name=dataset_name_2,
        )


def get_classification_label_to_id(
    config: PretrainedConfig, target_label_values: TYPE_DATASET_LABELS, is_regression: bool
) -> Dict[Union[str, int], int]:
    label_to_id: Dict[Union[str, int], int] = {}

    if is_regression:
        return label_to_id

    label_to_id = deepcopy(config.label2id)
    for i, label in enumerate(target_label_values):
        config.label2id[str(label)] = i
        label_to_id[str(label)] = i
        config.label2id[str(i)] = i
        label_to_id[str(i)] = i
        label_to_id[i] = i

    return label_to_id


def merge_and_trim_single_classification_datasets(
    dataset_list: List[LoadedDatasetsMappingSingle],
    target_labels: TYPE_DATASET_LABELS,
    max_samples: Optional[int],
    merge_datasets: bool,
    set_name: str,
) -> LoadedDatasetsMappingSingle:
    if len(dataset_list) > 1:
        LOGGER.info(f"{set_name} set contains {len(dataset_list)} datasets")

        if merge_datasets:
            LOGGER.info(f"Merging {len(dataset_list)} datasets in {set_name} set")
            # Check keys and labels are the same
            check_datasets_keys_and_labels(dataset_list, set_name)
            # Trim each dataset separated to make sure each dataset is in final dataset
            if max_samples is not None:
                max_samples_per_dataset = max_samples // len(dataset_list)
                LOGGER.info(
                    f"Limiting each dataset to {max_samples_per_dataset} samples, where max samples: {max_samples}"
                    f" and number datasets: {len(dataset_list)}"
                )
                total_samples = 0
                for dataset_index in range(len(dataset_list)):
                    dataset_1 = dataset_list[dataset_index]
                    dataset_1.dataset = trim_dataset(dataset_1.dataset, target_labels, max_samples_per_dataset)
                    total_samples += len(dataset_1.dataset)
                LOGGER.info(f"Finally, total samples is {total_samples}")

            dataset = dataset_list[0]
            dataset.dataset = _concatenate_map_style_datasets(
                [i_dataset.dataset for i_dataset in dataset_list], info=dataset.dataset.info
            )
            return dataset

        LOGGER.info(f"{set_name} set will use first dataset (found {len(dataset_list)} datasets)")
    # Use first dataset
    dataset = dataset_list[0]
    # Limit samples in needed
    if max_samples is not None:
        LOGGER.info(f"{set_name} set will be limited to {max_samples} samples")
        dataset.dataset = trim_dataset(dataset.dataset, target_labels, max_samples)

    return dataset


def merge_and_trim_classification_datasets(
    dataset_arguments: DataTrainingArguments,
    train_dataset_list: List[LoadedDatasetsMappingSingle],
    validation_dataset_list: List[LoadedDatasetsMappingSingle],
    target_labels: TYPE_DATASET_LABELS,
    dataset_name: str,
) -> Tuple[LoadedDatasetsMappingSingle, LoadedDatasetsMappingSingle]:
    train_dataset = merge_and_trim_single_classification_datasets(
        train_dataset_list,
        target_labels,
        dataset_arguments.max_train_samples,
        dataset_arguments.merge_datasets,
        f"{dataset_name}-train",
    )
    validation_dataset = merge_and_trim_single_classification_datasets(
        validation_dataset_list,
        target_labels,
        dataset_arguments.max_validation_samples,
        dataset_arguments.merge_datasets,
        f"{dataset_name}-validation",
    )
    return train_dataset, validation_dataset


def get_classification_dataset(dataset_args: DataTrainingArguments) -> Tuple[Dataset, Dataset, DatasetConfiguration]:
    # Load train and validation data
    loaded_datasets_mapping_train = load_dataset_dict_from_raw_data_file(dataset_args.train_file)
    loaded_datasets_mapping_validation = load_dataset_dict_from_raw_data_file(dataset_args.validation_file)
    # Get dataset data
    dataset_name = dataset_args.dataset_name
    train_dataset_list = loaded_datasets_mapping_train.dataset_name_to_dataset_data[dataset_name]
    train_dataset: LoadedDatasetsMappingSingle = train_dataset_list[0]
    validation_dataset_list = loaded_datasets_mapping_validation.dataset_name_to_dataset_data[dataset_name]
    validation_dataset: LoadedDatasetsMappingSingle = validation_dataset_list[0]
    # Check dataset metrics
    check_dataset_metrics(train_dataset.metric, validation_dataset.metric)
    # Make sure train and validation data have the same keys and labels
    train_text_key_names = get_dataset_text_keys(f"{dataset_name}-train", train_dataset.dataset)
    valid_text_key_names = get_dataset_text_keys(f"{dataset_name}-validation", validation_dataset.dataset)
    check_dataset_keys(train_text_key_names, valid_text_key_names)

    train_dataset_labels = get_labels_from_dataset(f"{dataset_name}-train", train_dataset.dataset)
    valid_dataset_labels = get_labels_from_dataset(f"{dataset_name}-validation", train_dataset.dataset)
    check_dataset_labels(train_dataset_labels, valid_dataset_labels)

    train_dataset, validation_dataset = merge_and_trim_classification_datasets(
        dataset_args, train_dataset_list, validation_dataset_list, train_dataset_labels, dataset_name
    )

    dataset_configuration = DatasetConfiguration(
        dataset_name=train_dataset.dataset_info.dataset_name,
        configuration_name=train_dataset.dataset_info.configuration_name,
        set_name=train_dataset.dataset_info.set_name,
        input_labels=train_text_key_names,
        target_label=TARGET_LABEL,
        target_label_values=train_dataset_labels,
        metric_configuration=train_dataset.metric,
    )

    return train_dataset.dataset, validation_dataset.dataset, dataset_configuration


def get_test_classification_data(
    dataset_args: DataEvaluationArguments,
) -> Tuple[List[LoadedDatasetsMappingSingle], DatasetConfiguration]:
    # Load test data
    loaded_datasets_mapping_test = load_dataset_dict_from_raw_data_file(dataset_args.test_file)
    # Get dataset data
    dataset_name = dataset_args.dataset_name
    test_dataset_list = loaded_datasets_mapping_test.dataset_name_to_dataset_data[dataset_name]
    test_dataset: LoadedDatasetsMappingSingle = test_dataset_list[0]
    # Get text and target keys (column names)
    test_text_key_names = get_dataset_text_keys(f"{dataset_name}-test", test_dataset.dataset)
    test_dataset_labels = get_labels_from_dataset(f"{dataset_name}-test", test_dataset.dataset)
    # Check keys and labels are the same
    check_datasets_keys_and_labels(test_dataset_list, dataset_name)

    dataset_configuration = DatasetConfiguration(
        dataset_name=test_dataset.dataset_info.dataset_name,
        configuration_name=test_dataset.dataset_info.configuration_name,
        set_name=test_dataset.dataset_info.set_name,
        input_labels=test_text_key_names,
        target_label=TARGET_LABEL,
        target_label_values=test_dataset_labels,
        metric_configuration=test_dataset.metric,
    )

    return test_dataset_list, dataset_configuration


def preprocess_classification_data(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    sequence_length: int,
    padding_strategy: Union[str, bool],
    label_to_id: Dict[Union[str, int], int],
    is_regression: bool,
    text_key_names: List[str],
    dataset_name: str = "",
    seed: Optional[int] = None,
) -> Dataset:
    def preprocess_fn(examples: LazyBatch) -> BatchEncoding:
        result = tokenizer(
            *[examples[text_key] for text_key in text_key_names],
            padding=padding_strategy,
            max_length=sequence_length,
            truncation=True,
        )
        if not is_regression:
            # Map labels to numbers
            result[TARGET_LABEL] = [
                (label_to_id[label_index] if label_index != -1 else -1) for label_index in examples[TARGET_LABEL]
            ]
        return result

    # Remove all not required columns after preprocessing
    columns_to_remove = list({feature for feature in list(dataset.features.keys()) if feature not in {TARGET_LABEL}})

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


def get_classification_metrics_fn(
    metric_configuration: MetricConfiguration, is_regression: bool
) -> Callable[[EvalPrediction], Dict[str, Any]]:
    LOGGER.info(f"Loading metric: {metric_configuration}")
    metric = evaluate.load(metric_configuration.name, config_name=metric_configuration.configuration_name)

    def compute_metrics(p: EvalPrediction) -> Dict[str, Any]:
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
        result: Dict[str, Any] = metric.compute(predictions=predictions, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    return compute_metrics
