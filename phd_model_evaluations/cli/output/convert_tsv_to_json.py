#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.output.convert_tsv_to_json \
  --input_file - --output_file - < glue-lm-gap/out-dev-0.tsv > results/glue/glue-lm-gap/validation.json
or:
python -m phd_model_evaluations.cli.output.convert_tsv_to_json \
  --input_file glue-lm-gap/out-dev-0.tsv --output_file results/glue/glue-lm-gap/validation.json
"""


import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.output.convert_tsv_to_json_arguments import ConvertTSVtoJSONArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.output.convert_tsv_to_json import convert_tsv_to_json, get_input_text
from phd_model_evaluations.utils.common_utils import save_json_data, sort_json_data

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[ConvertTSVtoJSONArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (ConvertTSVtoJSONArguments, LoggerArguments), description="Convert TSV data format to JSON data format"
    )
    conversion_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return conversion_args, logger_args


def main() -> None:
    conversion_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    input_text = get_input_text(conversion_args.input_file)
    json_data = convert_tsv_to_json(input_text)
    if isinstance(json_data, List) and conversion_args.sort_key_name is not None:
        json_data = sort_json_data(json_data, conversion_args.sort_key_name)
    save_json_data(json_data, conversion_args.output_file)


if __name__ == "__main__":
    main()
