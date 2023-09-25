from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DrawPlotArguments:
    """Arguments for drawing plot."""

    aggregated_results_path: Path = field(metadata={"help": "Aggregate results data path."})
    save_path: Path = field(metadata={"help": "Plot save path."})
    x_name: str = field(metadata={"help": "Name of column from table data which will be X axis."})
    y_names: List[str] = field(metadata={"help": "Name of columns from table data which will be Y axis."})
    key_names: Optional[List[str]] = field(default=None, metadata={"help": "Name of keys which will be used in plot."})
    x_title: Optional[str] = field(default=None, metadata={"help": "Name of X axis."})
    y_title: Optional[str] = field(default=None, metadata={"help": "Name of Y axis."})
    figure_height: int = field(default=12, metadata={"help": "Figure height in inches."})
    figure_width: int = field(default=12, metadata={"help": "Figure width in inches."})
    legend_columns: int = field(default=1, metadata={"help": "Number of columns in the legend."})
    skip_outside_points: bool = field(default=False, metadata={"help": "Skip outside points."})
    keep_min_outside_x_points: bool = field(
        default=False,
        metadata={"help": "Keep outside points below than the mean value with standard deviation in X axis."},
    )
    keep_max_outside_x_points: bool = field(
        default=False,
        metadata={"help": "Keep outside points above than the mean value with standard deviation in X axis."},
    )
    keep_min_outside_y_points: bool = field(
        default=False,
        metadata={"help": "Keep outside points below than the mean value with standard deviation in Y axis."},
    )
    keep_max_outside_y_points: bool = field(
        default=False,
        metadata={"help": "Keep outside points above than the mean value with standard deviation in Y axis."},
    )
    max_x_value: Optional[float] = field(default=None, metadata={"help": "Maximum value for X axis."})
    min_x_value: Optional[float] = field(default=None, metadata={"help": "Minimum value for X axis."})
    max_y_value: Optional[float] = field(default=None, metadata={"help": "Maximum value for Y axis."})
    min_y_value: Optional[float] = field(default=None, metadata={"help": "Minimum value for Y axis."})
    add_linear_regression: bool = field(default=False, metadata={"help": "Draw linear regression base on points."})
    add_label_at_top: bool = field(default=False, metadata={"help": "Draw labels on top of plot."})
