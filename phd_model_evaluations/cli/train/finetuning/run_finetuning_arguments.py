from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataTrainingArguments:
    """Arguments for training dataset configuration."""

    dataset_name: str = field(metadata={"help": "Name of dataset from train and validation datasets data."})
    train_file: Path = field(metadata={"help": "A file containing the training data."})
    validation_file: Path = field(metadata={"help": "A file containing the validation data."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples in train set, if missing, it will use all train data."},
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples in train set, if missing, it will use all validation data."},
    )
    merge_datasets: bool = field(
        default=True,
        metadata={"help": "Merge datasets data if is more than one dataset, otherwise will use first dataset data."},
    )
    saving_and_evaluation_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": 'Frequency ratio of saving and evaluation in one epoch. It will use "step" strategy for saving'
            " and evaluation. Number of steps will be computed automatically base on loaded dataset."
            ' For example if ratio is set to "0.1" it will save and evaluate on every 10%% of the epoch.'
        },
    )
    logging_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": 'Frequency ratio of logging in one epoch. It will use "step" strategy for logging.'
            " Number of steps will be computed automatically base on loaded dataset."
            " This works similar as --saving_and_evaluation_ratio argument."
        },
    )

    def __post_init__(self) -> None:
        self.check_saving_and_evaluation_ratio()
        self.check_logging_ratio()

    def check_saving_and_evaluation_ratio(self) -> None:
        if self.saving_and_evaluation_ratio is not None and (
            self.saving_and_evaluation_ratio <= 0.0 or self.saving_and_evaluation_ratio >= 1.0
        ):
            raise RuntimeError("Argument --saving_and_evaluation_ratio should be value between (0.0; 1.0)")

    def check_logging_ratio(self) -> None:
        if self.logging_ratio is not None and (self.logging_ratio <= 0.0 or self.logging_ratio >= 1.0):
            raise RuntimeError("Argument --logging_ratio should be value between (0.0; 1.0)")
