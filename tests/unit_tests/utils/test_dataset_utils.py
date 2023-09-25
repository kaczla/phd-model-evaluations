import pytest

from phd_model_evaluations.utils.dataset_utils import is_test_set_name, is_validation_set_name


@pytest.mark.parametrize("set_name", ["validation", "validation_matched", "validation_mismatched"])
def test_is_validation_set_name(set_name: str) -> None:
    assert is_validation_set_name(set_name), "Should be validation set name"


@pytest.mark.parametrize("set_name", ["train", "training", "test", "test_matched", "test_mismatched", "testing"])
def test_is_not_validation_set_name(set_name: str) -> None:
    assert not is_validation_set_name(set_name), "Should not be validation set name"


@pytest.mark.parametrize("set_name", ["test", "test_matched", "test_mismatched", "testing"])
def test_is_test_set_name(set_name: str) -> None:
    assert is_test_set_name(set_name), "Should be test set name"


@pytest.mark.parametrize("set_name", ["train", "training", "validation", "validation_matched", "validation_mismatched"])
def test_is_not_test_set_name(set_name: str) -> None:
    assert not is_test_set_name(set_name), "Should be test set name"
