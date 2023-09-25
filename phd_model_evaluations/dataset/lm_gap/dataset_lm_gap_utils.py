import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, List, Tuple

from phd_model_evaluations.data.lm_gap.lm_gap_dataset import LMGapDataset
from phd_model_evaluations.data.lm_gap.lm_gap_dataset_data import LMGapDatasetData
from phd_model_evaluations.data.saved_dataset_data import SavedDatasetsData
from phd_model_evaluations.dataset.lm_gap.dataset_lm_gap_string import (
    CONFIG_TEXT,
    GIT_IGNORE_TEXT,
    HEADER_IN_TEXT,
    HEADER_OUT_TEXT,
    README_TEXT,
)

LOGGER = logging.getLogger(__name__)


def get_lm_gap_file_path(set_path: Path, prefix_file_name: str) -> Path:
    for file_extension in ["", ".gz", ".xz"]:
        file_path = set_path / f"{prefix_file_name}{file_extension}"
        if file_path.exists():
            return file_path

    raise RuntimeError(f"Cannot find file with prefix: {prefix_file_name} in {set_path} set")


def get_dataset_data_for_directory_name(directory_name: str, dataset: LMGapDataset) -> List[LMGapDatasetData]:
    if directory_name in {"dev-0"}:
        return dataset.valid

    elif directory_name in {"test-A"}:
        return dataset.test

    elif directory_name in {"train"}:
        return dataset.train

    raise RuntimeError(f"Unknown dataset name: {directory_name}")


def get_set_names_form_directory_names(directory_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    train_set_names, valid_set_names, test_set_names = [], [], []

    for directory_name in directory_names:
        if directory_name.startswith("train"):
            train_set_names.append(directory_name)
        elif directory_name.startswith("dev-"):
            valid_set_names.append(directory_name)
        elif directory_name.startswith("test-"):
            test_set_names.append(directory_name)

    train_set_names.sort()
    valid_set_names.sort()
    test_set_names.sort()

    return train_set_names, valid_set_names, test_set_names


def generate_statists_readme(data_statistics: Dict[str, Counter]) -> str:
    # Get label/directory names
    train_set_names, valid_set_names, test_set_names = get_set_names_form_directory_names(list(data_statistics.keys()))
    set_names = [*train_set_names, *valid_set_names, *test_set_names]

    # Aggregate by dataset names
    data_set_names = set()
    aggregated_statistics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for set_name, set_statistics in data_statistics.items():
        for dataset_name, occurrences in set_statistics.items():
            data_set_names.add(dataset_name)
            aggregated_statistics[dataset_name][set_name] += occurrences

    # Create simple table:
    # | Dataset name | train | dev-0 | test-A |
    # | --- | --- | --- | --- |
    # | glue-cola | 7_240 | 804 | 967 |
    # ...
    # | **Total examples** | 1_597_709 | 177_521 | 135_612 |
    map_set_name_to_total_examples = {set_name: 0 for set_name in set_names}
    text = "| Dataset name | " + " | ".join(set_names) + " |\n"
    text += "| --- | " + " | ".join("---" for _ in range(len(set_names))) + " |\n"
    for dataset_name in sorted(aggregated_statistics.keys()):
        text += f"| {dataset_name} |"
        statistics = aggregated_statistics[dataset_name]
        for set_name in set_names:
            occurrences = statistics.get(set_name, 0)
            map_set_name_to_total_examples[set_name] += occurrences
            text += f" {occurrences:_d} |" if occurrences else " - |"
        text += "\n"

    text += "| **Total examples** |"
    for set_name in set_names:
        total_examples = map_set_name_to_total_examples[set_name]
        text += f" **{total_examples:_d}** |"
    text += "\n"

    return f"# Data statistics\n\nTable represents number of lines in each dataset:\n\n{text}\n"


def save_datasets_data(save_file_path: Path, raw_data: List[SavedDatasetsData]) -> None:
    if not raw_data:
        return

    with save_file_path.open("wt") as f_data:
        data_to_save = [data.dict() for data in raw_data]
        json.dump(data_to_save, f_data, indent=4, ensure_ascii=False)


def save_datasets(save_path: Path, datasets: List[LMGapDataset]) -> None:
    LOGGER.info(f"Saving datasets into: {save_path}")

    saved_data_statistics: Dict[str, Counter] = {}
    for set_name in ["train", "dev-0", "test-A"]:
        dir_path = save_path / set_name
        dir_path.mkdir(exist_ok=True)

        raw_data_path = dir_path / "raw_data.txt"
        input_path = dir_path / "in.tsv"
        expected_path = dir_path / "expected.tsv"
        dataset_data_path = dir_path / "dataset_data.json"

        all_raw_data = []
        saved_data_counter: CounterType = Counter()

        LOGGER.info(f"Saving into: {input_path}, {expected_path} and {raw_data_path} files")
        with input_path.open("wt") as f_in, raw_data_path.open("wt") as f_raw, expected_path.open("wt") as f_exp:
            for dataset in datasets:
                dataset_data_list = get_dataset_data_for_directory_name(set_name, dataset)
                for dataset_data in dataset_data_list:
                    for lm_gap_line in dataset_data.lm_gap_lines:
                        dataset_data_name = dataset_data.get_name()
                        saved_data_counter.update([dataset_data_name])
                        f_raw.write(f"{lm_gap_line.text}\n")
                        dataset_name_with_sub_split_name = dataset_data.get_name_with_sub_split_name()
                        f_in.write(
                            f"{dataset_name_with_sub_split_name}\t"
                            f"{lm_gap_line.left_context}\t"
                            f"{lm_gap_line.right_context}\n"
                        )
                        f_exp.write(f"{lm_gap_line.gap}\n")
                    if dataset_data.raw_data is not None:
                        all_raw_data.append(
                            SavedDatasetsData(
                                dataset_name=dataset_data.get_dataset_info(),
                                metric=dataset.metric,
                                feature_definitions=dataset_data.feature_definitions,
                                data=dataset_data.raw_data,
                            )
                        )

        save_datasets_data(dataset_data_path, all_raw_data)
        saved_data_statistics[set_name] = saved_data_counter

    save_meta_files(save_path, saved_data_statistics)


def save_meta_files(save_path: Path, data_statistics: Dict[str, Counter]) -> None:
    for file_name, file_content in [
        ("in-header.tsv", HEADER_IN_TEXT),
        ("out-header.tsv", HEADER_OUT_TEXT),
        ("config.txt", CONFIG_TEXT),
        ("README.md", README_TEXT + generate_statists_readme(data_statistics)),
        (".gitignore", GIT_IGNORE_TEXT),
    ]:
        file_path = save_path / file_name
        file_path.write_text(file_content)
