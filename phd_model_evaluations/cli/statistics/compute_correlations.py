#!/usr/bin/env python3

"""
Example run:
python -m phd_model_evaluations.cli.statistics.compute_correlation \
  --models_dir_path results/glue/finetuned_model/model_results.json \
  --save_path results/glue/finetuned_model/correlations.json
"""

import logging
from typing import Dict, List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.statistics.compute_correlations_arguments import ComputeCorrelationArguments
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.data.results.table_data import TableData
from phd_model_evaluations.data.statistics.correlations.computed_correlations import ComputedCorrelations
from phd_model_evaluations.statistics.compute_correlations import add_previous_correlations, compute_correlations
from phd_model_evaluations.utils.common_utils import load_json_data, round_float, save_json_data

LOGGER = logging.getLogger(__name__)

DEFAULT_X_LABELS = ["CoLA", "MNLI-m", "MNLI-mm", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B", "AVG"]
DEFAULT_Y_LABELS = ["loss", "LM-GAP"]


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[ComputeCorrelationArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (ComputeCorrelationArguments, LoggerArguments), description="Compute correlation of models results."
    )
    correlation_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return correlation_args, logger_args


def get_column_names(add_previous_data: bool) -> List[str]:
    if add_previous_data:
        return [
            "Nazwa zbioru",
            "Poprzedni współczynnik korelacji Pearsona",
            "Współczynnik korelacji Pearsona",
            "p-value ze współczynnika korelacji Pearsona",
            "Poprzedni współczynnik korelacji Spearmana",
            "Współczynnik korelacji Spearmana",
            "p-value ze współczynnika korelacji Spearmana",
        ]

    return [
        "Nazwa zbioru",
        "Współczynnik korelacji Pearsona",
        "p-value ze współczynnika korelacji Pearsona",
        "Współczynnik korelacji Spearmana",
        "p-value ze współczynnika korelacji Spearmana",
    ]


def convert_correlations_to_table_data(
    correlations: List[ComputedCorrelations], correlation_precision: int = 4, add_previous_data: bool = False
) -> TableData:
    column_names = get_column_names(add_previous_data)
    row_names = []
    row_data: List[Dict[str, str]] = []
    for computed_correlations in correlations:
        data_set_name = computed_correlations.x_name
        row_names.append(data_set_name)
        data: Dict[str, str] = {}
        for name, correlation, previous_correlation in [
            (
                "Pearsona",
                computed_correlations.pearson_correlation,
                computed_correlations.previous_pearson_correlation,
            ),
            (
                "Spearmana",
                computed_correlations.spearman_correlation,
                computed_correlations.previous_spearman_correlation,
            ),
        ]:
            correlation_value_str = str(round_float(correlation.correlation, precision=correlation_precision))
            if add_previous_data:
                row_name = f"Poprzedni współczynnik korelacji {name}"
                data[row_name] = (
                    "-"
                    if previous_correlation is None
                    else str(round_float(previous_correlation.correlation, precision=correlation_precision))
                )
                # Arrow up/down to show difference between values
                if previous_correlation is not None and correlation.correlation != previous_correlation.correlation:
                    arrow_str = "↑" if correlation.correlation > previous_correlation.correlation else "↓"
                    correlation_value_str += f" {arrow_str}"

            row_name = f"Współczynnik korelacji {name}"
            data[row_name] = correlation_value_str
            row_name = f"p-value ze współczynnika korelacji {name}"
            data[row_name] = "-" if correlation.p_value is None else f"{correlation.p_value:.2e}"

        row_data.append(data)

    return TableData(
        column_names=column_names,
        row_names=row_names,
        row_data=row_data,
        one_line_row_names=[],
        skip_row_name=False,
    )


def main() -> None:
    correlation_args, logger_args = parse_args()
    set_logging_from_logger_arguments(logger_args)

    aggregated_results: AggregatedResults = AggregatedResults(**load_json_data(correlation_args.file_path))
    if correlation_args.only_encoder or correlation_args.only_decoder or correlation_args.only_encoder_decoder:
        aggregated_results.filter_model_names(
            correlation_args.only_encoder, correlation_args.only_decoder, correlation_args.only_encoder_decoder
        )

    # Compute correlations
    correlations = compute_correlations(
        aggregated_results,
        correlation_args.x_labels if correlation_args.x_labels else DEFAULT_X_LABELS,
        correlation_args.y_labels if correlation_args.y_labels else DEFAULT_Y_LABELS,
        correlation_args.x_higher_better,
        correlation_args.y_higher_better,
    )

    # Add previous correlations
    add_previous_data = False
    if correlation_args.previous_correlations_file_path is not None:
        LOGGER.info(f"Reading previous correlations from: {correlation_args.previous_correlations_file_path}")
        add_previous_data = True
        previous_correlations: List[ComputedCorrelations] = [
            ComputedCorrelations(**single_correlation)
            for single_correlation in load_json_data(correlation_args.previous_correlations_file_path)
        ]
        correlations = add_previous_correlations(correlations, previous_correlations)

    # Save data
    save_json_data([c.dict() for c in correlations], correlation_args.save_path)
    if correlation_args.save_table_data:
        table_save_path = correlation_args.save_path.parent / ("table-" + correlation_args.save_path.name)
        save_json_data(
            convert_correlations_to_table_data(correlations, add_previous_data=add_previous_data).dict(),
            table_save_path,
        )


if __name__ == "__main__":
    main()
