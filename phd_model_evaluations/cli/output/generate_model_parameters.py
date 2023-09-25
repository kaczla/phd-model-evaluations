#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.output.generate_models_parameters \
  --save_path results/model_parameters.json
"""


import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.output.generate_model_parameters_arguments import GenerateModelParametersArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.output.generate_model_parameters import (
    convert_model_parameters_to_table_data,
    generate_model_parameters,
)
from phd_model_evaluations.utils.common_utils import save_json_data
from phd_model_evaluations.utils.model_names import DEFAULT_MODEL_NAMES

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[GenerateModelParametersArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (GenerateModelParametersArguments, LoggerArguments), description="Generate model parameters"
    )
    generate_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return generate_args, logger_args


def main() -> None:
    generate_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    model_names = DEFAULT_MODEL_NAMES if generate_args.model_names is None else generate_args.model_names
    models_parameters_list = generate_model_parameters(model_names)

    save_json_data(
        {models_parameters.name: models_parameters.dict() for models_parameters in models_parameters_list},
        generate_args.save_path,
    )

    if generate_args.save_table_data:
        table_save_path = generate_args.save_path.parent / ("table-" + generate_args.save_path.name)
        table_data = convert_model_parameters_to_table_data(models_parameters_list)
        save_json_data(table_data.dict(), table_save_path)


if __name__ == "__main__":
    main()
