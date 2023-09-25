#!/usr/bin/env python3

"""
Example run:
CUDA_VISIBLE_DEVICES='0' TRANSFORMERS_CACHE='.cache/transformers' \
python -m phd_model_evaluations.cli.statistics.lm_gap.generate_output_statistics \
  --path dev-0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.statistics.lm_gap.generate_output_statistics_arguments import (
    LMGapOutputStatisticsArguments,
)
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.lm_gap.lm_gap_output_statistics import LMGapOutputStatistics
from phd_model_evaluations.statistics.lm_gap.output_statistics import generate_lm_gap_output_statists

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[LMGapOutputStatisticsArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (LMGapOutputStatisticsArguments, LoggerArguments),
        description="Generate statistics for LM-GAP output predictions."
        " It will process all file starts with `out` in the name."
        " Script expect also `expected.tsv` to prediction matches.",
    )
    statistics_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return statistics_args, logger_args


def save_statistics(save_path: Path, statistics_dict: Dict[str, LMGapOutputStatistics]) -> None:
    LOGGER.info(f"Saving output statistics in: {save_path}")
    save_path.write_text(
        json.dumps(
            {name: statistics.dict() for name, statistics in statistics_dict.items()}, ensure_ascii=False, indent=4
        )
    )


def main() -> None:
    statistics_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    statistics_dict = generate_lm_gap_output_statists(statistics_args.path)

    save_path = statistics_args.path / "statistics-outputs.json"
    save_statistics(save_path, statistics_dict)


if __name__ == "__main__":
    main()
