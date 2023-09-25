#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.output.generate_aggregated_results \
  --results_file results/glue/finetuned_model/raw_model_results.json \
  --lm_gap_file results/glue/glue-lm-gap/validation.json \
  --loss_file results/glue/loss/validation.json \
  --save_path results/glue/finetuned_model/model_results.json
"""


import logging
from typing import List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.output.generate_aggregated_results_arguments import GenerateAggregatedResultsArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.data.results.lm_gap_results import LMGapResults
from phd_model_evaluations.data.statistics.loss_statistics import LossStatistics
from phd_model_evaluations.output.generate_aggregated_results import (
    compute_average_score,
    generate_aggregated_results_table,
)
from phd_model_evaluations.utils.common_utils import load_json_data, save_json_data

LOGGER = logging.getLogger(__name__)


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[GenerateAggregatedResultsArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (GenerateAggregatedResultsArguments, LoggerArguments), description="Generate aggregate results"
    )
    generate_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return generate_args, logger_args


def main() -> None:
    generate_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    results_data: AggregatedResults = AggregatedResults(**load_json_data(generate_args.results_file_path))
    lm_gap_data_list: List[LMGapResults] = (
        [
            LMGapResults(model_name=single_data["model_name"], score=single_data[generate_args.lm_gap_score_key_name])
            for single_data in load_json_data(generate_args.lm_gap_file_path)
        ]
        if generate_args.lm_gap_file_path is not None
        else []
    )
    loss_data_list: List[LossStatistics] = (
        [LossStatistics(**single_data) for single_data in load_json_data(generate_args.loss_file_path)]
        if generate_args.loss_file_path is not None
        else []
    )

    if generate_args.add_average_score:
        results_data = compute_average_score(results_data, score_precision=generate_args.score_precision)
    aggregated_results = generate_aggregated_results_table(
        results_data, lm_gap_data_list, loss_data_list, score_precision=generate_args.score_precision
    )

    save_json_data(aggregated_results.dict(), generate_args.save_path)
    if generate_args.save_table_data:
        table_save_path = generate_args.save_path.parent / ("table-" + generate_args.save_path.name)
        save_json_data(
            aggregated_results.get_table_data(return_empty=generate_args.return_empty_date).dict(), table_save_path
        )


if __name__ == "__main__":
    main()
