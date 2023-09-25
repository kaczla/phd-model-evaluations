#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.output.generate_latex_table \
  --table_path results/glue/model_validation_results.json \
  --save_path results/glue/model_validation_results.tex
"""


import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.output.generate_latex_table_arguments import GenerateLatexTableArguments
from phd_model_evaluations.data.arguments.latex_table_arguments import LatexTableArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.results.table_data import TableData
from phd_model_evaluations.output.generate_latex_table import generate_latex_table
from phd_model_evaluations.utils.common_utils import load_json_data

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[GenerateLatexTableArguments, LatexTableArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (GenerateLatexTableArguments, LatexTableArguments, LoggerArguments), description="Generate Latex table"
    )
    generate_args, table_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return generate_args, table_args, logger_args


def main() -> None:
    generate_args, table_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    table_data = TableData(**load_json_data(generate_args.table_path))
    generate_latex_table(table_data, generate_args.save_path, table_args)


if __name__ == "__main__":
    main()
