from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ComputeCorrelationArguments:
    """Arguments for compute correlation."""

    file_path: Path = field(metadata={"help": "JSON file path with aggregated results."})
    save_path: Path = field(metadata={"help": "Save path of computed correlations."})
    previous_correlations_file_path: Optional[Path] = field(
        default=None,
        metadata={"help": "JSON file path with previous correlations (which will be added to computed correlations)."},
    )
    save_table_data: bool = field(
        default=False,
        metadata={
            "help": 'Additional save table data as JSON, it will be saved with prefix "table" name'
            " in the same directory as save path."
        },
    )
    x_labels: Optional[List[str]] = field(
        default=None, metadata={"help": "Name of X labels used in computing correlations."}
    )
    y_labels: Optional[List[str]] = field(
        default=None, metadata={"help": "Name of Y labels used in computing correlations."}
    )
    x_higher_better: bool = field(
        default=True,
        metadata={
            "help": "X values higher values are better values (when values increase then are the better values),"
            " otherwise lower values are better, e.g. Accuracy is the higher values (higher value is the better score),"
            " loss is the lower values (lower value is the better score), expected X values as Benchmark score"
        },
    )
    y_higher_better: bool = field(
        default=False,
        metadata={
            "help": "Y values higher values are better values (when values increase then are the better values),"
            " otherwise lower values are better, e.g. Accuracy is the higher values (higher value is the better score),"
            " loss is the lower values (lower value is the better score), expected Y values as loss or LM-GAP score"
        },
    )
    only_encoder: bool = field(default=False, metadata={"help": "Compute correlations for encoder Transformer models"})
    only_decoder: bool = field(default=False, metadata={"help": "Compute correlations for decoder Transformer models"})
    only_encoder_decoder: bool = field(
        default=False, metadata={"help": "Compute correlations for encoder-decoder Transformer models"}
    )
