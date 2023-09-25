from typing import Dict, List

from pydantic import BaseModel


class TableData(BaseModel):
    column_names: List[str]
    row_names: List[str]
    # Row data: column name with value
    row_data: List[Dict[str, str]]
    one_line_row_names: List[str] = []
    skip_row_name: bool = False
