from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GenerateAggregatedLMGapResultsArguments:
    """Arguments for aggregating LM-GAP results into one file."""

    source_file_path: Path = field(metadata={"help": "A path to original LM-GAP results."})
    other_file_paths: List[Path] = field(
        metadata={"help": "A file paths to other LM-GAP results (which will be compared with the original results)."}
    )
    save_path: Path = field(metadata={"help": "A file path to save aggregated LM-GAP results."})
    save_table_data: bool = field(
        default=False,
        metadata={
            "help": 'Additional save table data as JSON, it will be saved with prefix "table" name'
            " in the same directory as save path."
        },
    )
    generate_in_groups: bool = field(
        default=False,
        metadata={"help": "Generate in groups (each group in separated file result) to avoid one huge table."},
    )
    name_source_file: str = field(default="LM-GAP", metadata={"help": "Name of source data file."})
    name_other_files: Optional[List[str]] = field(default=None, metadata={"help": "Name of other data files."})
    score_precision: int = field(
        default=2, metadata={"help": "Precision of scores, `None` means return original precision."}
    )
