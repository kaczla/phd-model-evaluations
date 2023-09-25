from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataEvaluationArguments:
    """Arguments for evaluate model configuration."""

    dataset_name: str = field(metadata={"help": "Name of dataset from test datasets data."})
    test_file: Path = field(metadata={"help": "A file containing the test data."})
    find_best_model: bool = field(
        default=False,
        metadata={
            "help": "Find best model in given directory base on metric evaluation (in `trainer_state.json` file)."
        },
    )
