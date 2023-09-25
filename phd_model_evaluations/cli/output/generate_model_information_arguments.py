from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GenerateModelInformationArguments:
    """Arguments for generating model information."""

    save_path: Path = field(metadata={"help": "Save path of model information."})
    model_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Name of models to generate list, otherwise will use default defined model."},
    )
    save_table_data: bool = field(
        default=False,
        metadata={
            "help": 'Additional save table data as JSON, it will be saved with prefix "table" name'
            " in the same directory as save path."
        },
    )
