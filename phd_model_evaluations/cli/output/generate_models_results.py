#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.output.generate_models_results \
  --models_dir_path out/glue/finetuned_model \
  --save_path results/glue/finetuned_model/raw_model_results.json
"""


import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.output.generate_models_results_arguments import GenerateModelsResultsArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.output.generate_models_results import generate_models_results
from phd_model_evaluations.utils.common_utils import save_json_data
from phd_model_evaluations.utils.model_names import DEFAULT_MODEL_HUMAN_NAMES

LOGGER = logging.getLogger(__name__)

DEFAULT_SEARCH_DATASET_NAMES = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
]


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[GenerateModelsResultsArguments, LoggerArguments]:
    parser = HfArgumentParser((GenerateModelsResultsArguments, LoggerArguments), description="Generate models results")
    generate_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return generate_args, logger_args


def main() -> None:
    generate_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    model_names = DEFAULT_MODEL_HUMAN_NAMES if generate_args.model_names is None else generate_args.model_names
    search_dataset_names = (
        DEFAULT_SEARCH_DATASET_NAMES if generate_args.dataset_names is None else generate_args.dataset_names
    )
    models_results = generate_models_results(
        generate_args.models_dir_path,
        model_names,
        search_dataset_names,
        return_empty_score=generate_args.return_empty_date,
        score_factor=generate_args.score_factor,
    )

    save_json_data(models_results.dict(), generate_args.save_path)


if __name__ == "__main__":
    main()
