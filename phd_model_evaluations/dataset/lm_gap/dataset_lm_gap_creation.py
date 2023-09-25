from random import randint
from typing import List

from phd_model_evaluations.data.lm_gap.lm_gap_dataset import LMGapDataset
from phd_model_evaluations.data.lm_gap.lm_gap_dataset_data import LMGapDatasetData
from phd_model_evaluations.data.lm_gap.lm_gap_line import LMGapLine
from phd_model_evaluations.data.loaded_dataset import LoadedDataset
from phd_model_evaluations.data.loaded_dataset_data import LoadedDatasetData
from phd_model_evaluations.utils.dataset_utils import DATASET_NAME_TO_CONFIGS, DATASET_NAME_TO_NAMES, load_dataset_data
from phd_model_evaluations.utils.regex_utils import REGEX_WHITESPACES, REGEX_WHITESPACES_AT_LEAST_TWO
from phd_model_evaluations.utils.type_utils import TYPE_DATASET_ELEMENT_LIST_DICT


def validate_lm_gap_text(text: str, left_context: str, right_context: str, gap: str) -> None:
    validate_text = f"{left_context} {gap} {right_context}".strip()
    if text != validate_text:
        raw_text = repr(text)
        raw_gap = repr(gap)
        raise RuntimeError(f"Invalid LM-GAP data for text: {raw_text} and gap: {raw_gap}")


def get_lm_gap_line(text: str, validate: bool = True) -> LMGapLine:
    text = text.strip().replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = REGEX_WHITESPACES.sub(" ", text)
    text = REGEX_WHITESPACES_AT_LEAST_TWO.sub(" ", text)

    tokens = list(filter(None, text.split()))

    gap_index = randint(0, len(tokens) - 1)  # noqa: S311

    left_context = " ".join(tokens[:gap_index])
    right_context = " ".join(tokens[gap_index + 1 :])
    gap = tokens[gap_index]

    if gap_index == 0:
        left_context = ""
    elif gap_index == len(tokens) - 1:
        right_context = ""

    lm_gap_line = LMGapLine(text=text, left_context=left_context, right_context=right_context, gap=gap)

    if validate:
        validate_lm_gap_text(text, left_context, right_context, gap)

    return lm_gap_line


def convert_dataset_data_to_lm_gap(dataset_data: LoadedDatasetData, validate: bool = True) -> LMGapDatasetData:
    lm_gap_lines = []
    raw_data: TYPE_DATASET_ELEMENT_LIST_DICT = {key: [] for key in dataset_data.get_raw_data_keys()}

    for data_text in dataset_data.data_text:
        lm_gap_lines.append(get_lm_gap_line(data_text.text, validate=validate))
        for key, value in data_text.raw_data.items():
            raw_data[key].append(value)

    return LMGapDatasetData(
        dataset_name=dataset_data.dataset_name,
        configuration_name=dataset_data.configuration_name,
        set_name=dataset_data.set_name,
        lm_gap_lines=lm_gap_lines,
        raw_data=raw_data,
        feature_definitions=dataset_data.feature_definitions,
    )


def convert_dataset_to_lm_gap(dataset: LoadedDataset, validate: bool = True) -> LMGapDataset:
    return LMGapDataset(
        train=[convert_dataset_data_to_lm_gap(dataset_data, validate=validate) for dataset_data in dataset.train],
        valid=[convert_dataset_data_to_lm_gap(dataset_data, validate=validate) for dataset_data in dataset.valid],
        test=[convert_dataset_data_to_lm_gap(dataset_data, validate=validate) for dataset_data in dataset.test],
        metric=dataset.metric,
    )


def get_lm_gap_datasets(
    dataset_name: str, join_sample_text: bool = False, skip_source_test_set: bool = False
) -> List[LMGapDataset]:
    loaded_datasets = []

    # Load list of datasets base on single dataset name
    if dataset_name in DATASET_NAME_TO_NAMES:
        for sub_dataset_name in DATASET_NAME_TO_NAMES[dataset_name]:
            loaded_data = load_dataset_data(
                sub_dataset_name, join_sample_text=join_sample_text, skip_source_test_set=skip_source_test_set
            )
            loaded_datasets.append(convert_dataset_to_lm_gap(loaded_data))
        return loaded_datasets

    list_configs = DATASET_NAME_TO_CONFIGS.get(dataset_name)

    # Load single dataset
    if list_configs is None:
        loaded_data = load_dataset_data(
            dataset_name, join_sample_text=join_sample_text, skip_source_test_set=skip_source_test_set
        )
        loaded_datasets.append(convert_dataset_to_lm_gap(loaded_data))

    # Load list of datasets (config) from single dataset name
    else:
        for config_name in list_configs:
            loaded_data = load_dataset_data(
                dataset_name,
                config_name=config_name,
                join_sample_text=join_sample_text,
                skip_source_test_set=skip_source_test_set,
            )
            loaded_datasets.append(convert_dataset_to_lm_gap(loaded_data))

    return loaded_datasets
