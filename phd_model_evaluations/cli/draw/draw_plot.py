#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.drwa.draw_plot \
  --aggregated_results_path results/glue/model_validation_results.json \
  --save_path results/glue/plot.png \
  --x_name loss \
  --y_names CoLA MRPC QNLI \
  --key_names RoBERTa-base RoBERTa-large
"""


import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.draw.draw_plot_arguments import DrawPlotArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.draw.draw_plot import draw_plot
from phd_model_evaluations.utils.common_utils import load_json_data

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[DrawPlotArguments, LoggerArguments]:
    parser = HfArgumentParser((DrawPlotArguments, LoggerArguments), description="Draw plot")
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

    table_data = AggregatedResults(**load_json_data(draw_args.aggregated_results_path))
    draw_plot(table_data, draw_args)


if __name__ == "__main__":
    main()
