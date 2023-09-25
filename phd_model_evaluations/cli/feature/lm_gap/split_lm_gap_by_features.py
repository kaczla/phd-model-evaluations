#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.dataset.create_lm_gap_challenge import check_save_path_exists
from phd_model_evaluations.cli.feature.lm_gap.split_lm_gap_by_features_arguments import SplitLMGapByFeaturesArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.dataset.lm_gap.dataset_lm_gap_utils import get_lm_gap_file_path
from phd_model_evaluations.dataset.lm_gap.features.feature_loader import (
    filter_features,
    get_features_from_lm_gap_set,
    get_lm_gap_feature_checkers,
    save_selected_lm_gap_set,
)

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[SplitLMGapByFeaturesArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (SplitLMGapByFeaturesArguments, LoggerArguments),
        description="Split LM-GAP set by features (like length of text, length of masked token).",
    )
    split_lm_gap_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return split_lm_gap_args, logger_args


def get_input_and_expected_file(set_path: Path) -> Tuple[Path, Path]:
    return get_lm_gap_file_path(set_path, "in.tsv"), get_lm_gap_file_path(set_path, "expected.tsv")


def split_lm_gap_by_features_main(cmd_args: Optional[List[str]] = None) -> None:
    split_lm_gap_args, logger_args = parse_args(cmd_args=cmd_args)
    set_logging_from_logger_arguments(logger_args)

    save_path = split_lm_gap_args.save_path
    input_file_path, expected_file_path = get_input_and_expected_file(split_lm_gap_args.set_path)
    check_save_path_exists(save_path, split_lm_gap_args.overwrite)
    # Get all LM-GAP features
    features_lm_gap = get_features_from_lm_gap_set(
        input_file_path,
        expected_file_path,
        split_lm_gap_args.column_left_context,
        split_lm_gap_args.column_right_context,
        skip_cache=split_lm_gap_args.skip_cache,
    )
    # Get LM-GAP feature checker
    lm_gap_feature_checkers = get_lm_gap_feature_checkers(split_lm_gap_args.checkers_file_path)
    name_to_checker = {checker.name: checker for checker in lm_gap_feature_checkers.checkers}
    # Filter with LM-GAP feature checkers
    features_lm_gap, selected_lines = filter_features(features_lm_gap, name_to_checker, split_lm_gap_args.set_path.name)

    LOGGER.info(f"Saving split LM-GAP into: {save_path}")
    save_path.mkdir(exist_ok=True)
    save_selected_lm_gap_set(
        input_file_path,
        expected_file_path,
        split_lm_gap_args.save_path,
        features_lm_gap,
        selected_lines,
        lm_gap_feature_checkers,
        save_all_predictions=not split_lm_gap_args.skip_all_predictions,
    )


if __name__ == "__main__":
    split_lm_gap_by_features_main()
