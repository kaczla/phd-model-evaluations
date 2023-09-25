from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AggregateLMGapFeaturesArguments:
    """Arguments for aggregating LM-GAP features (like length of text, length of masked token)."""

    input_file_path: Path = field(metadata={"help": "A file path to LM-GAP input file (in.tsv file)."})
    expected_file_path: Path = field(metadata={"help": "A file path to LM-GAP expected file (expected.tsv file)."})
    save_path: Path = field(metadata={"help": "A directory path to save aggregation of LM-GAP features."})
    column_left_context: int = field(
        default=1, metadata={"help": "Number of column of left context in input file, indexed from 0."}
    )
    column_right_context: int = field(
        default=2, metadata={"help": "Number of column of left context in input file, indexed from 0."}
    )
    skip_checking_input_directories: bool = field(
        default=False, metadata={"help": "Skip checking that input file and expected file have the same directory."}
    )
    skip_cache: bool = field(
        default=False,
        metadata={"help": "Skip saving cache for LM-GAP features, it will process all features from LM-GAP text."},
    )
    overwrite: bool = field(default=False, metadata={"help": "Overwrite data in save directory."})
