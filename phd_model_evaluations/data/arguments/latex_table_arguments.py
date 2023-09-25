import json
from dataclasses import dataclass, field
from typing import List, Optional

from phd_model_evaluations.data.latex.latex_table_type import LatexTableType


@dataclass
class LatexTableArguments:
    """Arguments for LaTeX table configuration."""

    label: Optional[str] = field(default=None, metadata={"help": "Table label."})
    caption: Optional[str] = field(default=None, metadata={"help": "Caption table."})
    table_type: LatexTableType = field(default=LatexTableType.TABLE, metadata={"help": "Type of table"})
    rotate_header_table: bool = field(default=False, metadata={"help": "Rotate header table text by 90 degrees."})
    mapping_label: str = field(
        default='{"AVG": "Średnia"}', metadata={"help": "Label/Column mapping as string Python dictionary."}
    )
    mapping_row_name: str = field(
        default='{"AVG": "Średnia"}', metadata={"help": "Row name (first column) mapping as string Python dictionary."}
    )
    mapping_parbox_size: str = field(
        default="{}",
        metadata={
            "help": "Mapping index header/column (indexing starts from 0) to parbox size in centimeters"
            ' as string Python dictionary, e.g. {"1": 3.5, "4": 5} will add parabox in column with index 1 and 4'
            " with value 3,5 cm and 5 cm respectively."
        },
    )
    selected_row_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of column names to select/return - other columns will be removed/skipped."},
    )

    def __post_init__(self) -> None:
        self.method = LatexTableType[str(self.table_type)]
        self.mapping_label_dict = json.loads(self.mapping_label)
        self.mapping_row_name_dict = json.loads(self.mapping_row_name)
        self.mapping_parabox_size_dict = {int(k): v for k, v in json.loads(self.mapping_parbox_size).items()}
        # Add names with "results" at the end
        if self.selected_row_names:
            for selected_row_name in self.selected_row_names.copy():
                self.selected_row_names.append(f"{selected_row_name} results")
