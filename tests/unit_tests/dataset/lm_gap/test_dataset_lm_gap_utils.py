import pytest

from phd_model_evaluations.data.lm_gap.lm_gap_dataset import LMGapDataset
from phd_model_evaluations.dataset.lm_gap.dataset_lm_gap_utils import get_dataset_data_for_directory_name


def test_get_dataset_data_for_directory_name_train(lm_gap_dataset: LMGapDataset) -> None:
    dataset_data = get_dataset_data_for_directory_name("train", lm_gap_dataset)
    assert len(dataset_data) == 1, "Expected 1 dataset data in train set"
    assert len(dataset_data[0].lm_gap_lines) == 2, "Expected 2 lines in train set"
    assert set(dataset_data[0].raw_data.keys()) == {"idx", "sentence"}, "Invalid keys in train raw data"
    assert dataset_data[0].raw_data["idx"] == [1, 5], "Invalid document indexes in train raw data"
    assert len(dataset_data[0].raw_data["sentence"]) == 2, "Invalid number of sentences in train raw data"


def test_get_dataset_data_for_directory_name_valid(lm_gap_dataset: LMGapDataset) -> None:
    dataset_data = get_dataset_data_for_directory_name("dev-0", lm_gap_dataset)
    assert len(dataset_data) == 1, "Expected 1 dataset data in validation set"
    assert len(dataset_data[0].lm_gap_lines) == 1, "Expected 1 lines in validation set"
    assert set(dataset_data[0].raw_data.keys()) == {"idx", "sentence"}, "Invalid keys in validation raw data"
    assert dataset_data[0].raw_data["idx"] == [2], "Invalid document indexes in train raw data"
    assert len(dataset_data[0].raw_data["sentence"]) == 1, "Invalid number of sentences in train raw data"


def test_get_dataset_data_for_directory_name_test(lm_gap_dataset: LMGapDataset) -> None:
    dataset_data = get_dataset_data_for_directory_name("test-A", lm_gap_dataset)
    assert len(dataset_data) == 1, "Expected 1 dataset data in test set"
    assert len(dataset_data[0].lm_gap_lines) == 3, "Expected 3 lines in test set"
    assert set(dataset_data[0].raw_data.keys()) == {"idx", "sentence"}, "Invalid keys in test raw data"
    assert dataset_data[0].raw_data["idx"] == [3, 4, 6], "Invalid document indexes in train raw data"
    assert len(dataset_data[0].raw_data["sentence"]) == 3, "Invalid number of sentences in train raw data"


def test_get_dataset_data_for_directory_name_exception(lm_gap_dataset: LMGapDataset) -> None:
    with pytest.raises(RuntimeError):
        get_dataset_data_for_directory_name("abc", lm_gap_dataset)
