import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, Features, load_dataset
from tqdm import tqdm

from phd_model_evaluations.data.loaded_dataset import LoadedDataset
from phd_model_evaluations.data.loaded_dataset_data import LoadedDatasetData
from phd_model_evaluations.data.loaded_dataset_mapping import LoadedDatasetsMapping
from phd_model_evaluations.data.loaded_dataset_mapping_single import LoadedDatasetsMappingSingle
from phd_model_evaluations.data.loaded_dataset_text_line import LoadedDatasetTextLine
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration
from phd_model_evaluations.data.saved_dataset_data import SavedDatasetsData
from phd_model_evaluations.utils.common_utils import get_open_fn
from phd_model_evaluations.utils.type_utils import TYPE_DATASET_LABELS

DATASET_NAME_TO_NAMES = {
    "klej": [
        "allegro/klej-nkjp-ner",
        "allegro/klej-allegro-reviews",
        "allegro/klej-cbd",
        "allegro/klej-cdsc-e",
        "allegro/klej-cdsc-r",
        "allegro/klej-dyk",
        "allegro/klej-polemo2-in",
        "allegro/klej-polemo2-out",
        "allegro/klej-psc",
    ],
}

DATASET_NAME_TO_CONFIGS = {
    "glue": ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"],
    "super_glue": ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc.fixed"],
}

DATASET_NAMES_TO_METRIC_CONFIGURATION: Dict[Tuple[str, ...], Dict[str, str]] = {
    ("glue", "cola"): {"name": "glue", "configuration_name": "cola"},
    ("glue", "sst2"): {"name": "glue", "configuration_name": "sst2"},
    ("glue", "mrpc"): {"name": "glue", "configuration_name": "mrpc"},
    ("glue", "qqp"): {"name": "glue", "configuration_name": "qqp"},
    ("glue", "stsb"): {"name": "glue", "configuration_name": "stsb"},
    ("glue", "mnli"): {"name": "glue", "configuration_name": "mnli"},
    ("glue", "qnli"): {"name": "glue", "configuration_name": "qnli"},
    ("glue", "rte"): {"name": "glue", "configuration_name": "rte"},
    ("glue", "wnli"): {"name": "glue", "configuration_name": "wnli"},
    ("glue", "hans"): {"name": "glue", "configuration_name": "hans"},
    ("super_glue", "axb"): {"name": "super_glue", "configuration_name": "axb"},
    ("super_glue", "axg"): {"name": "super_glue", "configuration_name": "axg"},
    ("super_glue", "boolq"): {"name": "super_glue", "configuration_name": "boolq"},
    ("super_glue", "cb"): {"name": "super_glue", "configuration_name": "cb"},
    ("super_glue", "copa"): {"name": "super_glue", "configuration_name": "copa"},
    ("super_glue", "multirc"): {"name": "super_glue", "configuration_name": "multirc"},
    ("super_glue", "record"): {"name": "super_glue", "configuration_name": "record"},
    ("super_glue", "rte"): {"name": "super_glue", "configuration_name": "rte"},
    ("super_glue", "wic"): {"name": "super_glue", "configuration_name": "wic"},
    ("super_glue", "wsc.fixed"): {"name": "super_glue", "configuration_name": "wsc.fixed"},
    ("allegro/klej-nkjp-ner",): {"name": "accuracy"},
    ("allegro/klej-allegro-reviews",): {"name": "mae"},
    ("allegro/klej-cbd",): {"name": "f1"},
    ("allegro/klej-cdsc-e",): {"name": "accuracy"},
    ("allegro/klej-cdsc-r",): {"name": "spearmanr"},
    ("allegro/klej-dyk",): {"name": "f1"},
    ("allegro/klej-polemo2-in",): {"name": "accuracy"},
    ("allegro/klej-polemo2-out",): {"name": "accuracy"},
    ("allegro/klej-psc",): {"name": "f1"},
}

DATASET_NAMES_TO_BEST_METRIC: Dict[Tuple[str, ...], str] = {
    ("glue", "cola"): "matthews_correlation",
    ("glue", "sst2"): "accuracy",
    ("glue", "mrpc"): "f1",
    ("glue", "qqp"): "f1",
    ("glue", "stsb"): "spearmanr",
    ("glue", "mnli"): "accuracy",
    ("glue", "mnli-m"): "accuracy",
    ("glue", "mnli-mm"): "accuracy",
    ("glue", "qnli"): "accuracy",
    ("glue", "rte"): "accuracy",
    ("glue", "wnli"): "accuracy",
    ("glue", "hans"): "accuracy",
    ("super_glue", "axb"): "matthews_correlation",
    ("super_glue", "axg"): "accuracy",
    ("super_glue", "boolq"): "accuracy",
    ("super_glue", "cb"): "f1",
    ("super_glue", "copa"): "accuracy",
    ("super_glue", "multirc"): "f1_a",
    ("super_glue", "record"): "f1",
    ("super_glue", "rte"): "accuracy",
    ("super_glue", "wic"): "accuracy",
    ("super_glue", "wsc.fixed"): "accuracy",
    ("allegro/klej-nkjp-ner",): "accuracy",
    ("allegro/klej-allegro-reviews",): "mae",
    ("allegro/klej-cbd",): "f1",
    ("allegro/klej-cdsc-e",): "accuracy",
    ("allegro/klej-cdsc-r",): "spearmanr",
    ("allegro/klej-dyk",): "f1",
    ("allegro/klej-polemo2-in",): "accuracy",
    ("allegro/klej-polemo2-out",): "accuracy",
    ("allegro/klej-psc",): "f1",
}

METRIC_NAME_TO_GREATER_IS_BETTER: Dict[str, bool] = {
    "loss": False,
    "matthews_correlation": True,
    "accuracy": True,
    "f1": True,
    "spearmanr": True,
    "pearson": True,
    "mae": True,
}

DATASET_PREDEFINED_FEATURES = [
    ["sentence"],
    ["sentence", "question"],
    ["question1", "question2"],
    ["sentence1", "sentence2"],
    ["premise", "hypothesis"],
    ["question", "passage"],
    ["premise", "choice1", "choice2", "question"],
    ["paragraph", "question", "answer"],
    ["passage", "query", "entities", "answers"],
]

DATASET_PREDEFINED_SET_NAMES = {
    "train",
    "validation",
    "test",
    "validation_matched",
    "validation_mismatched",
    "test_matched",
    "test_mismatched",
}

LOGGER = logging.getLogger(__name__)


def is_validation_set_name(name: str) -> bool:
    return "validation" in name


def is_test_set_name(name: str) -> bool:
    return "test" in name


def get_dataset_from_text(file_path: Path) -> Dataset:
    LOGGER.info(f"Loading text from file: {file_path}")
    open_fn = get_open_fn(file_path.name)

    dataset_json = []
    with open_fn(file_path, "rt") as f_read:
        for line in tqdm(f_read, desc="Reading text"):
            dataset_json.append({"text": line.strip()})

    LOGGER.info(f"Loaded {len(dataset_json)} lines")
    return Dataset.from_list(dataset_json)


def get_dataset_dict(dataset_name: str, config_name: Optional[str] = None) -> DatasetDict:
    if config_name is None:
        LOGGER.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        assert isinstance(dataset, DatasetDict), f"Expected DatasetDict type not: {type(dataset)}"
        return dataset

    LOGGER.info(f"Loading dataset: {dataset_name}-{config_name}")
    dataset = load_dataset(dataset_name, config_name)
    assert isinstance(dataset, DatasetDict), f"Expected DatasetDict type not: {type(dataset)}"
    return dataset


def get_dataset_metric(dataset_name: str, config_name: Optional[str] = None) -> MetricConfiguration:
    if config_name is not None:
        return MetricConfiguration(**DATASET_NAMES_TO_METRIC_CONFIGURATION[(dataset_name, config_name)])

    return MetricConfiguration(**DATASET_NAMES_TO_METRIC_CONFIGURATION[(dataset_name,)])


def get_metric_for_best_model(
    dataset_name: str,
    config_name: Optional[str],
    metric_for_best_model: Optional[str],
    greater_is_better: Optional[bool],
) -> Tuple[str, bool]:
    if metric_for_best_model is None:
        metric_name = (
            DATASET_NAMES_TO_BEST_METRIC[(dataset_name,)]
            if config_name is None
            else DATASET_NAMES_TO_BEST_METRIC[(dataset_name, config_name)]
        )
        greater_is_better = METRIC_NAME_TO_GREATER_IS_BETTER[metric_name]

    else:
        metric_name = metric_for_best_model
        if greater_is_better is None:
            greater_is_better = METRIC_NAME_TO_GREATER_IS_BETTER[metric_name]

    LOGGER.info(
        f"Using metric: {metric_name} (greater value is better"
        f" value: {greater_is_better}) for selecting best model."
    )
    return metric_name, greater_is_better


def get_dataset_feature_list(dataset: Dataset, expected_features_list: List[List[str]]) -> List[str]:
    data_features = set(dataset.features)

    # Remove unused features
    for feature in ["idx", "label"]:
        if feature in data_features:
            data_features.remove(feature)

    # Get expected feature list
    for features in expected_features_list:
        features_set = set(features)
        if features_set == data_features:
            return features

    raise RuntimeError(f"Cannot process: {dataset} - not supported feature names: {data_features}")


def get_text_data_from_dataset(dataset: Dataset, join_sample_text: bool = False) -> List[LoadedDatasetTextLine]:
    dataset_features = get_dataset_feature_list(dataset, DATASET_PREDEFINED_FEATURES)

    total_loaded_elements = 0
    text_data = []
    for data in dataset:
        text_list = [data[feature] for feature in dataset_features]

        # Join features to one text line
        if join_sample_text:
            text_list = [" ".join(text_list)]

        total_loaded_elements += len(text_list)

        for text in text_list:
            text = text.strip()

            # Accept text with at least 4 tokens
            if len(text.split()) <= 3:
                continue

            text_data.append(LoadedDatasetTextLine(text=text, raw_data=data))

    LOGGER.debug(f"Loaded {total_loaded_elements} lines, finally after filtering {len(text_data)} lines")

    return text_data


def split_validation_set_from_train_set(
    train_data: LoadedDatasetData, validation_ration: float = 0.1
) -> Tuple[LoadedDatasetData, LoadedDatasetData]:
    """
    Split train set into validation set and smaller train set.

    Args:
        train_data: train set
        validation_ration: ratio of data, what percentage of the text from train set should be in validation set,
            percentage is float value between (0.0, 1.0), default: is 10% of train set

    Returns:
        validation set and smaller train set

    """
    assert (
        0.0 < validation_ration < 1.0
    ), f"Ratio for validation set must be in (0.0, 1.0) range, not {validation_ration} value!"

    # Get balanced classes - for each class get given ration of data
    if (
        train_data
        and "idx" in train_data.data_text[0].raw_data
        and "label" in train_data.data_text[0].raw_data
        and isinstance(train_data.data_text[0].raw_data["label"], (int,))
    ):
        LOGGER.debug(f"Splitting with balanced classes with ratio {validation_ration} ...")

        # Get document indexes for each class/label
        map_class_index_to_example_indexes: Dict[int, List[str]] = defaultdict(lambda: list())
        for data in train_data.data_text:
            map_class_index_to_example_indexes[data.raw_data["label"]].append(data.raw_data["idx"])

        # Get document indexes for validation in each class/label
        validation_document_indexes = set()
        for key, values in map_class_index_to_example_indexes.items():
            total_examples = len(values)
            max_valid_examples = int(total_examples * validation_ration)
            validation_indexes = values[-max_valid_examples:]
            assert (
                len(validation_indexes) > 0
            ), f"Cannot split train set to validation set and train set for class: {key}"
            validation_document_indexes.update(validation_indexes)
            LOGGER.debug(
                f"Using {max_valid_examples} examples in {key} class (from {len(values)} elements) for validation set"
            )

        # Get validation and train examples
        split_train_data, split_valid_data = [], []
        for data in train_data.data_text:
            if data.raw_data["idx"] in validation_document_indexes:
                split_valid_data.append(data)
            else:
                split_train_data.append(data)

    else:
        LOGGER.debug(f"Splitting with ratio {validation_ration} ...")
        total_examples = len(train_data.data_text)
        max_valid_examples = int(total_examples * validation_ration)
        split_train_data = train_data.data_text[: total_examples - max_valid_examples]
        split_valid_data = train_data.data_text[-max_valid_examples:]
        assert len(split_valid_data) > 0, "Cannot split train set to validation set and train set"

    new_train_data = LoadedDatasetData(
        dataset_name=train_data.dataset_name,
        configuration_name=train_data.configuration_name,
        set_name=train_data.set_name + "[part-for-train]",
        data_text=split_train_data,
        feature_definitions=train_data.feature_definitions,
    )
    new_valid_data = LoadedDatasetData(
        dataset_name=train_data.dataset_name,
        configuration_name=train_data.configuration_name,
        set_name=train_data.set_name + "[part-for-validation]",
        data_text=split_valid_data,
        feature_definitions=train_data.feature_definitions,
    )
    LOGGER.info(
        f"Train set split from {len(train_data.data_text)} examples"
        f" to validation set with {len(new_valid_data.data_text)} examples"
        f" and new train set with {len(new_train_data.data_text)} examples"
    )

    assert len(train_data.data_text) == len(new_train_data.data_text) + len(new_valid_data.data_text), (
        f"Number of examples after train split should be the same!"
        f" Originally was {len(train_data.data_text)},"
        f" but after split is: {len(new_train_data.data_text) + len(new_valid_data.data_text)}"
    )
    return new_train_data, new_valid_data


def create_validation_and_test_set(dataset: LoadedDataset) -> LoadedDataset:
    """
    Set test set as validation set and create validation set from part of train set.

    Args:
        dataset: loaded dataset

    Returns:
        dataset with validation and test set

    """
    LOGGER.info("Setting test set from validation set")
    dataset.test = dataset.valid
    dataset.valid = []

    # Create validation set from part of train set
    LOGGER.info("Getting validation set from part of train set")
    new_train_data, new_valida_data = [], []
    for train_data in dataset.train:
        train, valid = split_validation_set_from_train_set(train_data)
        new_train_data.append(train)
        new_valida_data.append(valid)
    dataset.train = new_train_data
    dataset.valid = new_valida_data

    return dataset


def load_dataset_data(
    dataset_name: str,
    config_name: Optional[str] = None,
    join_sample_text: bool = False,
    skip_source_test_set: bool = False,
) -> LoadedDataset:
    dataset_dict = get_dataset_dict(dataset_name, config_name=config_name)
    metric = get_dataset_metric(dataset_name, config_name=config_name)
    loaded_dataset = LoadedDataset(train=[], valid=[], test=[], metric=metric)

    for set_name, dataset in dataset_dict.items():
        if set_name not in DATASET_PREDEFINED_SET_NAMES:
            LOGGER.warning(f"Skipped set name: {set_name}")
            continue

        if skip_source_test_set and "test" in set_name:
            LOGGER.debug(f"Skipping test set name: {set_name}")
            continue

        LOGGER.debug(f"Processing set name: {set_name}")
        data_text = get_text_data_from_dataset(dataset, join_sample_text=join_sample_text)
        feature_definitions = dataset.features.to_dict()

        if is_validation_set_name(set_name):
            loaded_dataset.valid.append(
                LoadedDatasetData(
                    dataset_name=dataset_name,
                    configuration_name=config_name,
                    set_name=set_name,
                    data_text=data_text,
                    feature_definitions=feature_definitions,
                )
            )
        elif is_test_set_name(set_name):
            loaded_dataset.test.append(
                LoadedDatasetData(
                    dataset_name=dataset_name,
                    configuration_name=config_name,
                    set_name=set_name,
                    data_text=data_text,
                    feature_definitions=feature_definitions,
                )
            )
        else:
            loaded_dataset.train.append(
                LoadedDatasetData(
                    dataset_name=dataset_name,
                    configuration_name=config_name,
                    set_name=set_name,
                    data_text=data_text,
                    feature_definitions=feature_definitions,
                )
            )

    if skip_source_test_set:
        loaded_dataset = create_validation_and_test_set(loaded_dataset)

    LOGGER.info(
        f"Train set loaded from {len(loaded_dataset.train)} datasets,"
        f" total loaded lines: {loaded_dataset.get_total_train_lines()}"
    )
    LOGGER.info(
        f"Validation set loaded from {len(loaded_dataset.valid)} datasets,"
        f" total loaded lines: {loaded_dataset.get_total_valid_lines()}"
    )
    LOGGER.info(
        f"Test set loaded from {len(loaded_dataset.test)} datasets,"
        f" total loaded lines: {loaded_dataset.get_total_test_lines()}"
    )
    return loaded_dataset


def load_dataset_dict_from_raw_data_file(file_path: Path) -> LoadedDatasetsMapping:
    LOGGER.info(f"Loading dataset data from: {file_path}")
    loaded_dataset_mapping = LoadedDatasetsMapping()
    open_fn = get_open_fn(file_path.name)
    with open_fn(file_path, "rt") as f_read:
        loaded_data = json.load(f_read)
        for loaded_data_dict in loaded_data:
            saved_datasets = SavedDatasetsData(**loaded_data_dict)
            dataset_info = saved_datasets.dataset_name
            dataset_name = (
                dataset_info.dataset_name
                if dataset_info.configuration_name is None
                else dataset_info.configuration_name
            )
            feature_definitions = Features.from_dict(saved_datasets.feature_definitions)
            dataset = Dataset.from_dict(saved_datasets.data, features=feature_definitions)
            loaded_dataset_mapping.dataset_name_to_dataset_data[dataset_name].append(
                LoadedDatasetsMappingSingle(dataset_info=dataset_info, dataset=dataset, metric=saved_datasets.metric)
            )
    LOGGER.info(
        f"Loaded {loaded_dataset_mapping.total_dataset()} datasets"
        f" and {loaded_dataset_mapping.total_dataset_data()} datasets data"
    )
    return loaded_dataset_mapping


def trim_dataset(dataset: Dataset, target_labels: TYPE_DATASET_LABELS, max_samples: int) -> Dataset:
    if len(dataset) <= max_samples:
        LOGGER.info(
            f"Dataset size ({len(dataset)} samples) is lower than expected size ({max_samples} samples),"
            f" skipping trimming"
        )
        return dataset

    if not target_labels:
        max_samples = min(len(dataset), max_samples)
        LOGGER.info(f"Trimmed with {max_samples} samples")
        dataset = dataset.select(range(max_samples))
        return dataset

    label_to_indexes = defaultdict(list)
    for i, sample in enumerate(dataset):
        label_to_indexes[sample["label"]].append(i)

    max_samples_per_label = int(max_samples / len(target_labels))
    sample_indexes = []
    for i in range(len(target_labels)):
        sample_indexes.extend(label_to_indexes[i][:max_samples_per_label])
    sample_indexes.sort()

    LOGGER.info(
        f"Trimmed with {max_samples_per_label} samples per label ({len(target_labels)} labels),"
        f" where total processed samples: {len(sample_indexes)} and max samples: {max_samples}"
    )
    dataset = dataset.select(sample_indexes)
    return dataset
