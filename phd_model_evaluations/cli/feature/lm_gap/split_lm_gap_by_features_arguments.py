from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SplitLMGapByFeaturesArguments:
    """Arguments for splitting LM-GAP set by features (like length of text, length of masked token)."""

    set_path: Path = field(metadata={"help": "A directory path to LM-GAP set (e.g. test-A)."})
    save_path: Path = field(metadata={"help": "A directory path to save split LM-GAP."})
    checkers_file_path: Optional[Path] = field(
        default=None, metadata={"help": "Path to LM-GAP feature checkers configurations."}
    )
    column_left_context: int = field(
        default=1, metadata={"help": "Number of column of left context in input file, indexed from 0."}
    )
    column_right_context: int = field(
        default=2, metadata={"help": "Number of column of left context in input file, indexed from 0."}
    )
    skip_cache: bool = field(
        default=False, metadata={"help": "Skip cached features, it will process all features from LM-GAP text."}
    )
    skip_all_predictions: bool = field(
        default=False, metadata={"help": "Skip all prediction outputs (out.tsv file/s)."}
    )
    overwrite: bool = field(default=False, metadata={"help": "Overwrite data in save directory."})
    dump_example_checkers_path: Optional[Path] = field(
        default=None,
        metadata={
            "help": "Path to example configuration of LM-GAP feature checkers,"
            " it will save example configuration for easier changing of LM-GAP feature checkers configuration."
        },
    )
