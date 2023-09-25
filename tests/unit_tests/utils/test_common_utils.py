from typing import Dict, List, Optional, Union

import pytest

from phd_model_evaluations.utils.common_utils import round_float, sort_json_data


@pytest.mark.parametrize(
    ("value", "precision", "expected_value"),
    [
        (10.0, None, 10.0),
        (123.0, None, 123.0),
        (123.4, None, 123.4),
        (123.45, None, 123.45),
        (123.456, None, 123.456),
        (123.4567, None, 123.4567),
        (10.0, 0, 10.0),
        (10.0, 1, 10.0),
        (10.0, 2, 10.0),
        (10.0, 3, 10.0),
        (123.4567, 1, 123.5),
        (123.4567, 2, 123.46),
        (123.4567, 3, 123.457),
        (123.4567, 4, 123.4567),
        (123.9999, 1, 124.0),
        (123.9999, 2, 124.0),
        (123.9999, 3, 124.0),
        (123.999, 4, 123.999),
        (999.999, 1, 1000.0),
        (999.999, 2, 1000.0),
        (123.4, 1, 123.4),
        (123.5, 1, 123.5),
        (123.6, 1, 123.6),
        (123.24, 1, 123.2),
        (123.25, 1, 123.2),
        (123.26, 1, 123.3),
        (123.224, 2, 123.22),
        (123.225, 2, 123.22),
        (123.226, 2, 123.23),
    ],
)
def test_round_float(value: float, precision: Optional[int], expected_value: float) -> None:
    result = round_float(value, precision)
    assert result == expected_value, "Invalid rounded float"


def test_sort_json_data() -> None:
    data: List[Dict[str, Union[int, float, str]]] = [
        {"test": "test", "name": "testing", "pi": 3.14},
        {"name": "error", "error": "some error"},
        {"name": "A1", "value": "same value", "a": 5, "b": 10},
    ]
    expected_data: List[Dict[str, Union[int, float, str]]] = [
        {"name": "A1", "value": "same value", "a": 5, "b": 10},
        {"name": "error", "error": "some error"},
        {"test": "test", "name": "testing", "pi": 3.14},
    ]
    result_data = sort_json_data(data, "name")
    assert expected_data == result_data, "Data after sorting are not equal"


def test_sort_json_data_exception() -> None:
    data: List[Dict[str, Union[int, float, str]]] = [
        {"test": "test", "name": "testing"},
        {"error": "some error"},
        {"name": "A1", "value": "same value"},
    ]
    result_data = sort_json_data(data, "name")
    assert data == result_data, "Data after sorting should be the same"
