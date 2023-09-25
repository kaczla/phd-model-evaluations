from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerateLatexTableArguments:
    """Arguments for generating Latex table from table JSON data."""

    table_path: Path = field(metadata={"help": "Table JSON data path"})
    save_path: Path = field(metadata={"help": "Save Latex file path."})
