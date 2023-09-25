from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LMGapFeaturesDrawPlotArguments:
    """Arguments for drawing LM-GAP features plot."""

    aggregated_features_path: Path = field(metadata={"help": "Aggregated LM-GAP features data path."})
    save_path: Path = field(metadata={"help": "Plot save path."})
    figure_height: int = field(default=12, metadata={"help": "Figure height in inches."})
    figure_width: int = field(default=12, metadata={"help": "Figure width in inches."})
    min_value: int = field(
        default=5, metadata={"help": "Min value to draw, otherwise will be ignored (only histogram)."}
    )
    overwrite: bool = field(default=False, metadata={"help": "Overwrite plots in save directory."})
