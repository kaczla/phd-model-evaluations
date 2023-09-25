from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConvertTSVtoJSONArguments:
    """Arguments for converting TSV to JSON."""

    input_file: Path = field(metadata={"help": "Input TSV file path."})
    output_file: Path = field(metadata={"help": "Output/Save JSON file path."})
    sort_key_name: Optional[str] = field(
        default=None, metadata={"help": "Name of key in the dictionary to sort list of dictionaries."}
    )
