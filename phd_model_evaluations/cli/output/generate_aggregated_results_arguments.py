from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LM_GAP_METRIC_KEY_NAME = "PerplexityHashed"


@dataclass
class GenerateAggregatedResultsArguments:
    """Arguments for generating aggregated results."""

    results_file_path: Path = field(metadata={"help": "JSON file path with challenges results."})
    save_path: Path = field(metadata={"help": "Save path of aggregate data."})
    lm_gap_file_path: Optional[Path] = field(
        default=None, metadata={"help": "JSON file path to evaluated LM-GAP challenges."}
    )
    loss_file_path: Optional[Path] = field(default=None, metadata={"help": "JSON file path with loss results."})
    lm_gap_score_key_name: str = field(
        default=LM_GAP_METRIC_KEY_NAME, metadata={"help": "Name of key for loading score for LM-GAP."}
    )
    score_precision: int = field(
        default=2, metadata={"help": "Precision of scores, `None` means return original precision."}
    )
    save_table_data: bool = field(
        default=False,
        metadata={
            "help": 'Additional save table data as JSON, it will be saved with prefix "table" name'
            " in the same directory as save path."
        },
    )
    add_average_score: bool = field(default=False, metadata={"help": "Add average for each data set split."})
    return_empty_date: bool = field(default=False, metadata={"help": "Return rows with empty data in table data."})
