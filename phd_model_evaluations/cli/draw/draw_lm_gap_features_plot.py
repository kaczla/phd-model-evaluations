#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.draw.draw_lm_gap_features_plot \
  --aggregated_features_path results/glue/filtered_by_features/aggregation.json \
  --save_path results/glue/filtered_by_features/plots
"""


import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.dataset.create_lm_gap_challenge import check_save_path_exists
from phd_model_evaluations.cli.draw.draw_lm_gap_features_plot_arguments import LMGapFeaturesDrawPlotArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.feature.aggregated_feature import AggregatedFeature
from phd_model_evaluations.draw.draw_lm_gap_feature_plot import draw_lm_gap_features_plot
from phd_model_evaluations.utils.common_utils import load_json_data

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[LMGapFeaturesDrawPlotArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (LMGapFeaturesDrawPlotArguments, LoggerArguments), description="Draw plot for LM-GAP features"
    )
    plot_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return plot_args, logger_args


def main() -> None:
    draw_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)
    if logger_args.verbose:
        loger = logging.getLogger("matplotlib")
        loger.setLevel(logging.INFO)
        loger = logging.getLogger("PIL")
        loger.setLevel(logging.INFO)

    check_save_path_exists(draw_args.save_path, draw_args.overwrite)

    aggregated_lm_gap_features = AggregatedFeature(**load_json_data(draw_args.aggregated_features_path))
    draw_args.save_path.mkdir(exist_ok=True)
    draw_lm_gap_features_plot(aggregated_lm_gap_features, draw_args)


if __name__ == "__main__":
    main()
