#!/usr/bin/env python3

import logging
from itertools import chain
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

from transformers import HfArgumentParser

from phd_model_evaluations.cli.output.lm_gap.generate_aggregated_lm_gap_results_arguments import (
    GenerateAggregatedLMGapResultsArguments,
)
from phd_model_evaluations.data.arguments.logger_arguments import LoggerArguments, set_logging_from_logger_arguments
from phd_model_evaluations.data.lm_gap.lm_gap_result import LMGapResult
from phd_model_evaluations.data.lm_gap.lm_gap_result_many import LMGapResultMany
from phd_model_evaluations.data.results.table_data import TableData
from phd_model_evaluations.utils.common_utils import load_json_data, round_float, save_json_data
from phd_model_evaluations.utils.translations import STRING_AVERAGE

LOGGER = logging.getLogger(__name__)

# Means: "is_number" is the same group as "gap_with_punctuation"
PREDEFINED_GROUP_NAMES_AS_SAME_GROUP = {"is_number": "gap_with_punctuation"}

PREDEFINED_ORDER = [
    "gap_with_punctuation",
    "is_number",
    "masked_token_frequency",
    "masked_token_length",
    "left_context_length",
    "right_context_length",
    "text_length",
]


def parse_args(cmd_args: Optional[List[str]] = None) -> Tuple[GenerateAggregatedLMGapResultsArguments, LoggerArguments]:
    parser = HfArgumentParser(
        (GenerateAggregatedLMGapResultsArguments, LoggerArguments),
        description="Aggregate LM-GAP results into one file.",
    )
    lm_gap_args, logger_args = parser.parse_args_into_dataclasses(args=cmd_args, look_for_args_file=False)
    return lm_gap_args, logger_args


def parse_lm_gap_results(file_path: Path, data_name: str) -> LMGapResultMany:
    lm_gap_results = [LMGapResult(**data) for data in load_json_data(file_path)]
    return LMGapResultMany(
        name=data_name,
        model_names=sorted(lm_gap_result.model_name for lm_gap_result in lm_gap_results),
        model_name_to_result={lm_gap_result.model_name: lm_gap_result for lm_gap_result in lm_gap_results},
    )


def parse_other_lm_gap_results(other_file_paths: List[Path], file_names: Optional[List[str]]) -> List[LMGapResultMany]:
    file_names = (
        file_names
        if file_names and len(file_names) == len(other_file_paths)
        else [other_file_path.name for other_file_path in other_file_paths]
    )
    assert len(file_names) == len(
        other_file_paths
    ), f"Invalid name of other LM-GAP result files - expected {len(other_file_paths)}, but got {len(file_names)}!"
    other_lm_gap_results_with_file_name = [
        parse_lm_gap_results(other_file_path, file_name)
        for file_name, other_file_path in zip(file_names, other_file_paths, strict=True)
    ]
    return other_lm_gap_results_with_file_name


def compute_average_lm_gap_result(model_name: str, lm_gap_results: Iterable[LMGapResult]) -> LMGapResult:
    return LMGapResult(
        model_name=model_name, PerplexityHashed=mean([lm_gap_result.score for lm_gap_result in lm_gap_results])
    )


def add_average_lm_gap_results(
    lm_gap_results: LMGapResultMany, other_lm_gap_results: List[LMGapResultMany]
) -> Tuple[LMGapResultMany, List[LMGapResultMany]]:
    for i_lm_gap_results in [lm_gap_results, *other_lm_gap_results]:
        i_lm_gap_results.model_names.append(STRING_AVERAGE)
        i_lm_gap_results.model_name_to_result[STRING_AVERAGE] = compute_average_lm_gap_result(
            STRING_AVERAGE, i_lm_gap_results.model_name_to_result.values()
        )
    return lm_gap_results, other_lm_gap_results


def convert_aggregated_lm_gap_results_to_table_data(
    lm_gap_results: LMGapResultMany, other_lm_gap_results: List[LMGapResultMany], score_precision: int = 4
) -> TableData:
    one_line_row_names = []
    model_names_set = set(lm_gap_results.model_names)
    model_names_set.update(chain.from_iterable(data.model_names for data in other_lm_gap_results))
    row_names = []
    # Row data = list of model names with dictionary, which maps dataset name to score
    row_data: List[Dict[str, str]] = []
    for model_name in sorted(model_names_set):
        row_names.append(model_name)
        one_line_row_names.append(model_name)
        row_data.append({})

        row_names.append(f"{model_name} results")
        # Add original LM-GAP results
        original_model_result = lm_gap_results.model_name_to_result.get(model_name)
        if original_model_result is None:
            LOGGER.error(f"Missing original LM-GAP result for model: {model_name}")
            row_data.append({})
            continue

        single_row_data = {
            lm_gap_results.name: str(round_float(original_model_result.score, precision=score_precision))
        }
        # Add other LM-GAP results
        for data in other_lm_gap_results:
            model_result = data.model_name_to_result.get(model_name)
            if model_result is not None:
                model_score = model_result.score
                value = str(round_float(model_score, precision=score_precision))
                if model_score != original_model_result.score:
                    arrow_str = "↑" if model_score < original_model_result.score else "↓"
                    value += f" {arrow_str}"

                single_row_data[data.name] = value

        row_data.append(single_row_data)

    return TableData(
        column_names=[lm_gap_results.name] + [data.name for data in other_lm_gap_results],
        row_names=row_names,
        row_data=row_data,
        one_line_row_names=one_line_row_names,
        skip_row_name=True,
    )


def get_group_names(other_lm_gap_results: List[LMGapResultMany]) -> List[List[str]]:
    name_to_normalized_name: Dict[str, str] = {
        result.name: result.name.rstrip("0123456789-_").rstrip() for result in other_lm_gap_results
    }

    normalized_name_to_grouped_names: Dict[str, List[str]] = {}
    for name, normalized_name in name_to_normalized_name.items():
        group_name = PREDEFINED_GROUP_NAMES_AS_SAME_GROUP.get(normalized_name, normalized_name)
        if group_name in normalized_name_to_grouped_names:
            normalized_name_to_grouped_names[group_name].append(name)
        else:
            normalized_name_to_grouped_names[group_name] = [name]

    # Keep predefined order of names
    result = []
    for name in PREDEFINED_ORDER:
        if name in normalized_name_to_grouped_names:
            result.append(normalized_name_to_grouped_names.pop(name))
    result.extend(sorted([names for _, names in sorted(normalized_name_to_grouped_names.items(), key=lambda x: x[0])]))

    return result


def save_lm_gap_results(
    lm_gap_args: GenerateAggregatedLMGapResultsArguments,
    lm_gap_results: LMGapResultMany,
    other_lm_gap_results: List[LMGapResultMany],
    group_names: Optional[List[str]] = None,
    group_id: int = 1,
) -> None:
    selected_other_lm_gap_results = []
    if group_names:
        group_names_set = set(group_names)
        save_path = lm_gap_args.save_path.with_stem(lm_gap_args.save_path.stem + f"-group_{group_id}")
        selected_other_lm_gap_results = [data for data in [*other_lm_gap_results] if data.name in group_names_set]
    else:
        save_path = lm_gap_args.save_path
    LOGGER.info(f"Saving aggregated LM-GAP features into: {save_path}")
    save_json_data([data.dict() for data in [lm_gap_results, *selected_other_lm_gap_results]], save_path)

    if lm_gap_args.save_table_data:
        table_save_path = save_path.parent / ("table-" + save_path.name)
        table_data = convert_aggregated_lm_gap_results_to_table_data(
            lm_gap_results, selected_other_lm_gap_results, score_precision=lm_gap_args.score_precision
        )
        save_json_data(table_data.dict(), table_save_path)


def save_lm_gap_results_in_groups(
    lm_gap_args: GenerateAggregatedLMGapResultsArguments,
    lm_gap_results: LMGapResultMany,
    other_lm_gap_results: List[LMGapResultMany],
) -> None:
    if lm_gap_args.generate_in_groups:
        group_names_list = get_group_names(other_lm_gap_results)
        for group_id, group_names in enumerate(group_names_list):
            save_lm_gap_results(
                lm_gap_args,
                lm_gap_results,
                other_lm_gap_results,
                group_names=group_names,
                group_id=group_id,
            )

    else:
        save_lm_gap_results(
            lm_gap_args,
            lm_gap_results,
            other_lm_gap_results,
        )


def aggregate_lm_gap_results_main(cmd_args: Optional[List[str]] = None) -> None:
    lm_gap_args, logger_args = parse_args(cmd_args=cmd_args)
    set_logging_from_logger_arguments(logger_args)
    # Get original and other LM-GAP results
    lm_gap_results = parse_lm_gap_results(lm_gap_args.source_file_path, lm_gap_args.name_source_file)
    other_lm_gap_results = parse_other_lm_gap_results(lm_gap_args.other_file_paths, lm_gap_args.name_other_files)
    # Add average LM-GAP results for each test set
    lm_gap_results, other_lm_gap_results = add_average_lm_gap_results(lm_gap_results, other_lm_gap_results)
    # Save results in groups (separated result file for each group) or in one result file
    save_lm_gap_results_in_groups(lm_gap_args, lm_gap_results, other_lm_gap_results)


if __name__ == "__main__":
    aggregate_lm_gap_results_main()
