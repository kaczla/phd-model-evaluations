from typing import List

import pytest

from phd_model_evaluations.data.lm_gap.lm_gap_dataset import LMGapDataset
from phd_model_evaluations.data.lm_gap.lm_gap_dataset_data import LMGapDatasetData
from phd_model_evaluations.data.lm_gap.lm_gap_line import LMGapLine
from phd_model_evaluations.data.metric.metric_configuration import MetricConfiguration


@pytest.fixture(scope="session")
def lm_gap_dataset_data_train() -> List[LMGapDatasetData]:
    return [
        LMGapDatasetData(
            dataset_name="testing",
            configuration_name=None,
            set_name="train",
            lm_gap_lines=[
                LMGapLine(
                    left_context="This",
                    right_context="a simple sentence.",
                    gap="is",
                    text="This is a simple sentence.",
                ),
                LMGapLine(
                    left_context="",
                    right_context="is the first day of the week.",
                    gap="Monday",
                    text="Monday is the first day of the week.",
                ),
            ],
            raw_data={
                "idx": [1, 5],
                "sentence": ["This is a simple sentence.", "Monday is the first day of the week."],
            },
            feature_definitions={
                "idx": {"dtype": "int32", "_type": "Value"},
                "sentence": {"dtype": "string", "_type": "Value"},
            },
        )
    ]


@pytest.fixture(scope="session")
def lm_gap_dataset_data_valid() -> List[LMGapDatasetData]:
    return [
        LMGapDatasetData(
            dataset_name="testing",
            configuration_name=None,
            set_name="validation",
            lm_gap_lines=[
                LMGapLine(left_context="cat.", right_context="Alice has", gap="a", text="Alice has a cat."),
            ],
            raw_data={
                "idx": [2],
                "sentence": ["Alice has a cat."],
            },
            feature_definitions={
                "idx": {"dtype": "int32", "_type": "Value"},
                "sentence": {"dtype": "string", "_type": "Value"},
            },
        )
    ]


@pytest.fixture(scope="session")
def lm_gap_dataset_data_test() -> List[LMGapDatasetData]:
    return [
        LMGapDatasetData(
            dataset_name="testing",
            configuration_name=None,
            set_name="test",
            lm_gap_lines=[
                LMGapLine(left_context="1 2 3", right_context="5 6 7", gap="4", text="1 2 3 4 5 6 7"),
                LMGapLine(
                    left_context="",
                    right_context="Second, Third, Fourth, Fifth.",
                    gap="First,",
                    text="First, Second, Third, Fourth, Fifth.",
                ),
                LMGapLine(
                    left_context="Another", right_context="sentence.", gap="random", text="Another random sentence."
                ),
            ],
            raw_data={
                "idx": [3, 4, 6],
                "sentence": ["1 2 3 4 5 6 7", "First, Second, Third, Fourth, Fifth.", "Another random sentence."],
            },
            feature_definitions={
                "idx": {"dtype": "int32", "_type": "Value"},
                "sentence": {"dtype": "string", "_type": "Value"},
            },
        )
    ]


@pytest.fixture(scope="session")
def lm_gap_dataset(
    lm_gap_dataset_data_train: List[LMGapDatasetData],
    lm_gap_dataset_data_valid: List[LMGapDatasetData],
    lm_gap_dataset_data_test: List[LMGapDatasetData],
) -> LMGapDataset:
    return LMGapDataset(
        train=lm_gap_dataset_data_train,
        valid=lm_gap_dataset_data_valid,
        test=lm_gap_dataset_data_test,
        metric=MetricConfiguration(name="accuracy"),
    )
