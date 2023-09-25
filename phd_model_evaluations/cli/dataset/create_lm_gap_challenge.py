#!/usr/bin/env python3

"""
Example run:
HF_DATASETS_CACHE=.cache_datasets \
python -m phd_model_evaluations.cli.dataset.create_lm_gap_challenge \
  --dataset_name glue \
  --save_path challenges/glue_lm_gap \
  --skip_source_test_set True
or:
HF_DATASETS_CACHE=.cache_datasets \
python -m phd_model_evaluations.cli.dataset.create_lm_gap_challenge \
  --dataset_name super_glue \
  --save_path challenges/super_glue_lm_gap \
  --skip_source_test_set True
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.dataset.create_lm_gap_challenge_arguments import LMGapChallengeArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.dataset.lm_gap.dataset_lm_gap_creation import get_lm_gap_datasets
from phd_model_evaluations.dataset.lm_gap.dataset_lm_gap_utils import save_datasets

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[LMGapChallengeArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (LMGapChallengeArguments, LoggerArguments),
        description="Create LM-GAP challenge base on HuggingFace datasets library.",
    )
    lm_gap_challenge_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return lm_gap_challenge_args, logger_args


def check_save_path_exists(save_path: Path, overwrite: bool = False) -> None:
    if save_path.exists():
        if not overwrite:
            raise RuntimeError(f'Path "{save_path}" exists!')
        elif not save_path.is_dir():
            raise RuntimeError(f'Path "{save_path}" is not a directory!')

    if not save_path.parent.exists():
        raise RuntimeError(f'Parent path "{save_path.parent}" does not exist')


def create_lm_gap_challenge_main(cmd_args: Optional[List[str]] = None) -> None:
    lm_gap_challenge_args, logger_args = parse_args(cmd_args=cmd_args)
    set_logging_from_logger_arguments(logger_args)

    save_path = lm_gap_challenge_args.save_path
    overwrite = lm_gap_challenge_args.overwrite
    check_save_path_exists(save_path, overwrite)

    random.seed(lm_gap_challenge_args.seed)
    loaded_datasets = get_lm_gap_datasets(
        lm_gap_challenge_args.dataset_name,
        join_sample_text=lm_gap_challenge_args.join_sample_text,
        skip_source_test_set=lm_gap_challenge_args.skip_source_test_set,
    )

    LOGGER.info(f"Created challenge directory: {save_path}")
    save_path.mkdir(exist_ok=overwrite)
    save_datasets(save_path, loaded_datasets)


if __name__ == "__main__":
    create_lm_gap_challenge_main()
