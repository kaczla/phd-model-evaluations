#!/usr/bin/env python3

import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.feature.lm_gap.dump_example_lm_gap_feature_checkers_arguments import (
    DumpExampleLMGapFeatureCheckersArguments,
)
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.dataset.lm_gap.features.feature_loader import dump_example_lm_gap_checkers

LOGGER = logging.getLogger(__name__)


def parse_args(
    cmd_args: Optional[List[str]] = None,
) -> Tuple[DumpExampleLMGapFeatureCheckersArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (DumpExampleLMGapFeatureCheckersArguments, LoggerArguments),
        description="Split LM-GAP set by features (like length of text, length of masked token).",
    )
    split_lm_gap_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return split_lm_gap_args, logger_args


def dump_example_lm_gap_feature_checkers(cmd_args: Optional[List[str]] = None) -> None:
    split_lm_gap_args, logger_args = parse_args(cmd_args=cmd_args)
    set_logging_from_logger_arguments(logger_args)

    dump_example_lm_gap_checkers(split_lm_gap_args.save_path)


if __name__ == "__main__":
    dump_example_lm_gap_feature_checkers()
