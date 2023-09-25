#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.dataset.create_lm_gap_challenge import check_save_path_exists
from phd_model_evaluations.cli.feature.lm_gap.aggregate_lm_gap_features_arguments import AggregateLMGapFeaturesArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.dataset.lm_gap.features.feature_loader import (
    aggregate_all_lm_gap_features,
    get_features_from_lm_gap_set,
    save_aggregated_lm_gap_features,
)

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[AggregateLMGapFeaturesArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (AggregateLMGapFeaturesArguments, LoggerArguments),
        description="Aggregate LM-GAP features (like length of text, length of masked token).",
    )
    lm_gap_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return lm_gap_args, logger_args


def check_input_and_expected_are_from_the_same_set(
    input_file_path: Path, expected_file_path: Path, skip_checking: bool
) -> None:
    if skip_checking:
        return

    if input_file_path.parent != expected_file_path.parent:
        raise RuntimeError(
            f"Input file ({input_file_path}) and expected file ({expected_file_path}) are from different directories!"
        )


def split_lm_gap_by_features_main(cmd_args: Optional[List[str]] = None) -> None:
    split_lm_gap_args, logger_args = parse_args(cmd_args=cmd_args)
    set_logging_from_logger_arguments(logger_args)

    save_path = split_lm_gap_args.save_path
    check_input_and_expected_are_from_the_same_set(
        split_lm_gap_args.input_file_path,
        split_lm_gap_args.expected_file_path,
        split_lm_gap_args.skip_checking_input_directories,
    )
    check_save_path_exists(save_path, split_lm_gap_args.overwrite)

    features_lm_gap = get_features_from_lm_gap_set(
        split_lm_gap_args.input_file_path,
        split_lm_gap_args.expected_file_path,
        split_lm_gap_args.column_left_context,
        split_lm_gap_args.column_right_context,
        skip_cache=split_lm_gap_args.skip_cache,
    )
    aggregated_lm_gap_features_list = aggregate_all_lm_gap_features(
        features_lm_gap.features, features_lm_gap.features[0].features
    )

    LOGGER.info(f"Saving aggregated LM-GAP features into: {save_path}")
    save_path.mkdir(exist_ok=True)
    save_aggregated_lm_gap_features(split_lm_gap_args.save_path, aggregated_lm_gap_features_list)


if __name__ == "__main__":
    split_lm_gap_by_features_main()
