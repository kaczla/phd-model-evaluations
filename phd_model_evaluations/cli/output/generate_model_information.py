#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.output.generate_models_information \
  --save_path results/model_information.json
"""

import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.output.generate_model_information_arguments import GenerateModelInformationArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.output.generate_model_information import (
    convert_model_information_to_table_data,
    generate_model_information,
)
from phd_model_evaluations.utils.common_utils import save_json_data
from phd_model_evaluations.utils.model_names import DEFAULT_MODEL_NAMES

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[GenerateModelInformationArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (GenerateModelInformationArguments, LoggerArguments), description="Generate model information"
    )
    generate_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return generate_args, logger_args


def main() -> None:
    generate_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    model_names = DEFAULT_MODEL_NAMES if generate_args.model_names is None else generate_args.model_names
    models_information_list = generate_model_information(model_names)

    save_json_data(
        {models_information.human_name: models_information.dict() for models_information in models_information_list},
        generate_args.save_path,
    )

    if generate_args.save_table_data:
        table_save_path = generate_args.save_path.parent / ("table-" + generate_args.save_path.name)
        table_data = convert_model_information_to_table_data(models_information_list)
        save_json_data(table_data.dict(), table_save_path)


if __name__ == "__main__":
    main()
