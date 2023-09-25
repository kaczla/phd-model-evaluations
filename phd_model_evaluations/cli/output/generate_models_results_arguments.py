from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GenerateModelsResultsArguments:
    """Arguments for generating models results."""

    models_dir_path: Path = field(
        metadata={
            "help": "Directory path where models are saved. Will be search model directories names as:"
            ' "MODEL_NAME-DATASET_NAME" where MODEL_NAME is name of model and DATASET_NAME is name of dataset.'
        }
    )
    save_path: Path = field(metadata={"help": "Save path of models data."})
    model_names: Optional[List[str]] = field(default=None, metadata={"help": "Name of models to generate results."})
    dataset_names: Optional[List[str]] = field(default=None, metadata={"help": "Name of dataset to generate results."})
    return_empty_date: bool = field(default=False, metadata={"help": "Return rows with empty data in table data."})
    score_factor: float = field(default=100.0, metadata={"help": "Score factor which will be multiplied with score."})
