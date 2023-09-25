import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

import regex
from tqdm import tqdm
from tqdm.contrib import tenumerate, tzip

from phd_model_evaluations.data.feature.aggregated_feature import AggregatedFeature
from phd_model_evaluations.data.feature.checker.base_feature import BaseFeature
from phd_model_evaluations.data.feature.checker.gap_with_punctuation_feature import GapWithPunctuationFeature
from phd_model_evaluations.data.feature.checker.is_number_feature import IsNumberFeature
from phd_model_evaluations.data.feature.checker.left_context_length_feature import LeftContextLengthFeature
from phd_model_evaluations.data.feature.checker.masked_token_frequency_feature import MaskedTokenFrequencyFeature
from phd_model_evaluations.data.feature.checker.masked_token_length_feature import MaskedTokenLengthFeature
from phd_model_evaluations.data.feature.checker.right_context_length_feature import RightContextLengthFeature
from phd_model_evaluations.data.feature.checker.text_length_feature import TextLengthFeature
from phd_model_evaluations.data.feature.feature_data import FeatureData
from phd_model_evaluations.data.feature.feature_statistics import FeatureStatistics
from phd_model_evaluations.data.feature.line_features import ALL_FEATURES_TYPE, LineFeatures
from phd_model_evaluations.data.feature.lm_gap_checker_utils import (
    ALL_FEATURE_CHECKERS_TYPE,
    LMGapFeatureCheckers,
    get_example_lm_gap_feature_checkers,
    parse_lm_gap_feature_checkers,
)
from phd_model_evaluations.data.feature.selected_lines import SelectedLines
from phd_model_evaluations.data.lm_gap.lm_gap_context_line_with_gap import LMGapContextLineWithGap
from phd_model_evaluations.evaluation.lm_gap.lm_gap_utils import load_lm_gap_context_lines_with_gap
from phd_model_evaluations.utils.common_utils import get_open_fn, load_json_data, save_json_data

LOGGER = logging.getLogger(__name__)

FEATURES_CACHE_FILE_NAME = ".features_cache"
FEATURES_CHECKERS_FILE_NAME = "lm_gap_feature_checkers.json"
SELECTED_LINES_FILE_NAME = "selected_lines.json"

RGX_GAP_END_WITH_PUNCTUATION = regex.compile(r"\p{Punct}$")
RGX_GAP_START_WITH_PUNCTUATION = regex.compile(r"^\p{Punct}")
RGX_GAP_WITH_ANY_PUNCTUATION = regex.compile(r"\p{Punct}")
RGX_GAP_IS_PUNCTUATION = regex.compile(r"^\p{Punct}$")
RGX_OUTPUT_FILE_NAME = regex.compile(r"^(?:out[.]tsv|out[-].+?[.]tsv)(?:[.]gz|[.]xz)?$")


def get_features_from_line(
    lm_gap_line: LMGapContextLineWithGap, line_id: int, feature_statistics: FeatureStatistics
) -> LineFeatures:
    features: List[ALL_FEATURES_TYPE] = [
        TextLengthFeature(
            text_length=sum(
                [len(text.split()) for text in [lm_gap_line.left_context, lm_gap_line.right_context, lm_gap_line.gap]]
            ),
        ),
        MaskedTokenLengthFeature(
            token_length=len(lm_gap_line.gap),
        ),
        LeftContextLengthFeature(
            left_context_length=len(lm_gap_line.left_context.split()),
        ),
        RightContextLengthFeature(
            right_context_length=len(lm_gap_line.right_context.split()),
        ),
        GapWithPunctuationFeature(
            end_with_punctuation=bool(RGX_GAP_END_WITH_PUNCTUATION.search(lm_gap_line.gap)),
            start_with_punctuation=bool(RGX_GAP_START_WITH_PUNCTUATION.search(lm_gap_line.gap)),
            has_punctuation=bool(RGX_GAP_WITH_ANY_PUNCTUATION.search(lm_gap_line.gap)),
            is_punctuation=bool(RGX_GAP_IS_PUNCTUATION.search(lm_gap_line.gap)),
        ),
        IsNumberFeature.from_gap(lm_gap_line.gap),
        MaskedTokenFrequencyFeature(
            masked_token=lm_gap_line.gap,
            masked_token_frequency=feature_statistics.token_frequency.get(lm_gap_line.gap, 0),
        ),
    ]
    return LineFeatures(line_id=line_id, features=features)


def load_feature_lm_gap_cache(cache_file_path: Path, skip_cache: bool = False) -> Optional[FeatureData]:
    if not cache_file_path.exists():
        return None
    elif skip_cache:
        LOGGER.debug(f"Skipping cache from: {cache_file_path}")
        return None

    LOGGER.info(f"Using cached features from: {cache_file_path}")
    return FeatureData(**load_json_data(cache_file_path))


def get_features_from_lm_gap_set(
    input_file_path: Path,
    expected_file_path: Path,
    column_left_context: int,
    column_right_context: int,
    skip_cache: bool = False,
) -> FeatureData:
    cache_file_path = input_file_path.parent / FEATURES_CACHE_FILE_NAME
    cache_lm_gap_features = load_feature_lm_gap_cache(cache_file_path, skip_cache=skip_cache)
    if cache_lm_gap_features is not None:
        return cache_lm_gap_features

    loaded_lm_gap_set = load_lm_gap_context_lines_with_gap(
        input_file_path, expected_file_path, column_left_context, column_right_context
    )

    line_features_list: List[LineFeatures] = []
    lm_gap_statistics = FeatureStatistics()
    for lm_gap_single_data in tqdm(loaded_lm_gap_set, desc="Computing statistics"):
        lm_gap_statistics.update_statistics(lm_gap_single_data)
    for line_index, lm_gap_single_data in tenumerate(loaded_lm_gap_set, desc="Computing features"):
        line_features_list.append(get_features_from_line(lm_gap_single_data, line_index, lm_gap_statistics))
    LOGGER.info(f"Computed {len(line_features_list)} LM-GAP features")

    lm_gap_statistics.aggregate_statistics()
    lm_gap_features = FeatureData(features=line_features_list, statistics=lm_gap_statistics)

    LOGGER.info(f"Saving LM-GAP features into cache: {cache_file_path}")
    save_json_data(lm_gap_features.dict(), cache_file_path)

    return lm_gap_features


def get_lm_gap_feature_checkers(file: Optional[Path]) -> LMGapFeatureCheckers:
    if file is None:
        LOGGER.info("LM-GAP feature checkers disabled")
        return LMGapFeatureCheckers(checkers=[])

    lm_gap_feature_checkers = parse_lm_gap_feature_checkers(load_json_data(file))
    LOGGER.info(f"Loaded {len(lm_gap_feature_checkers.checkers)} LM-GAP feature checkers")
    return lm_gap_feature_checkers


def filter_features(
    lm_gap_features: FeatureData, name_to_checker: Mapping[str, ALL_FEATURE_CHECKERS_TYPE], set_name: str
) -> Tuple[FeatureData, SelectedLines]:
    LOGGER.info("Filtering LM-GAP features...")
    filtered_line_features_list = []
    for line_features in tqdm(lm_gap_features.features, desc="Filtering LM-GAP lines with LM-GAP feature checkers"):
        if not name_to_checker or all(
            name_to_checker[feature.name].is_accepted_value(feature, lm_gap_features.statistics)
            for feature in line_features.features
            if feature.name in name_to_checker
        ):
            filtered_line_features_list.append(line_features)

    source_total_lines = len(lm_gap_features.features)
    total_lines = len(filtered_line_features_list)
    selected_lines = SelectedLines(
        source_set_name=set_name,
        source_total_lines=source_total_lines,
        total_lines=total_lines,
        total_lines_percentage=float(total_lines) / float(source_total_lines),
        indexes=[line_features.line_id for line_features in filtered_line_features_list],
    )

    LOGGER.info(
        f"Filtered {selected_lines.total_lines} LM-GAP lines from {selected_lines.source_total_lines} LM-GAP lines"
    )
    lm_gap_features.features = filtered_line_features_list
    return lm_gap_features, selected_lines


def aggregate_all_lm_gap_features(
    line_features_list: List[LineFeatures], features_to_aggregate: List[ALL_FEATURES_TYPE]
) -> List[AggregatedFeature]:
    LOGGER.info("Aggregating LM-GAP features")
    feature_name_to_features: Dict[str, List[BaseFeature]] = {
        feature.name: [] for feature in sorted(features_to_aggregate, key=lambda feature: feature.name)
    }
    for line_features in line_features_list:
        for feature in line_features.features:
            feature_name_to_features[feature.name].append(feature)

    aggregation = [
        features[0].aggregate(features, sort_data=True) for feature_name, features in feature_name_to_features.items()
    ]
    LOGGER.info("LM-GAP features aggregated")
    return aggregation


def dump_example_lm_gap_checkers(save_path: Path) -> None:
    example_lm_gap_feature_checkers = get_example_lm_gap_feature_checkers()
    LOGGER.info(f"Saving example configuration of LM-GAP feature checkers in: {save_path}")
    save_json_data(example_lm_gap_feature_checkers.dict(), save_path)


def save_lm_gap_prediction_file(prediction_file_path: Path, save_path: Path, accepted_line_indexes: Set[int]) -> None:
    open_fn = get_open_fn(prediction_file_path.name)
    with open_fn(prediction_file_path, "rt") as f_read, open_fn(save_path, "wt") as f_write:
        for line_index, line in enumerate(f_read):
            if line_index in accepted_line_indexes:
                f_write.write(line)


def save_lm_gap_predictions(input_directory_path: Path, save_path: Path, accepted_line_indexes: Set[int]) -> None:
    LOGGER.info("Saving predictions...")
    for path in sorted(input_directory_path.iterdir()):
        if not path.is_file() or not RGX_OUTPUT_FILE_NAME.search(path.name):
            continue

        LOGGER.info(f"Saving prediction for: {path.name} ...")
        save_file_path = save_path / path.name
        save_lm_gap_prediction_file(path, save_file_path, accepted_line_indexes)

    LOGGER.info("All predictions saved")


def save_selected_lm_gap_set(
    input_file_path: Path,
    expected_file_path: Path,
    save_path: Path,
    lm_gap_features: FeatureData,
    selected_lines: SelectedLines,
    lm_gap_feature_checkers: LMGapFeatureCheckers,
    save_all_predictions: bool = False,
) -> None:
    # Save LM-GAP feature checkers
    save_json_data(lm_gap_feature_checkers.dict(), save_path / FEATURES_CHECKERS_FILE_NAME)
    # Save line indexes
    save_json_data(selected_lines.dict(), save_path / SELECTED_LINES_FILE_NAME)
    # Save LM-GAP features cache
    save_json_data(lm_gap_features.dict(), save_path / FEATURES_CACHE_FILE_NAME)
    # Save input and expected data
    accepted_line_indexes = set(selected_lines.indexes)
    open_fn_input_file_path = get_open_fn(input_file_path.name)
    open_fn_expected_file_path = get_open_fn(expected_file_path.name)
    save_input_file_path = save_path / input_file_path.name
    save_expected_file_path = save_path / expected_file_path.name
    with (
        open_fn_input_file_path(input_file_path, "rt") as f_read_input,
        open_fn_expected_file_path(expected_file_path, "rt") as f_read_expected,
        open_fn_input_file_path(save_input_file_path, "wt") as f_write_input,
        open_fn_expected_file_path(save_expected_file_path, "wt") as f_write_expected,
    ):
        for line_index, (input_line, expected_line) in enumerate(
            tzip(f_read_input, f_read_expected, desc="Loading and saving text and gaps")
        ):
            if line_index in accepted_line_indexes:
                f_write_input.write(input_line)
                f_write_expected.write(expected_line)
    LOGGER.info("Input and expected file saved")

    if save_all_predictions:
        save_lm_gap_predictions(input_file_path.parent, save_path, accepted_line_indexes)


@lru_cache
def get_group_frequency_string(value: float, max_value_for_each_group: Tuple[int, ...]) -> Optional[str]:
    previous_value = 0
    for max_value in max_value_for_each_group:
        if value < max_value:
            return f"Liczba wystąpień w zakresie od {previous_value + 1} do {max_value}"
        previous_value = max_value

    return None


def convert_token_frequency_to_group_frequency(
    aggregated_lm_gap_features: AggregatedFeature, max_value_for_each_group: Tuple[int, ...]
) -> AggregatedFeature:
    LOGGER.info(
        f"Converting token frequency to group frequency for: {aggregated_lm_gap_features.name}"
        f" with max value for each group: {max_value_for_each_group}"
    )
    counter = Counter(
        (
            value
            for value in (
                get_group_frequency_string(value, max_value_for_each_group)
                for value in aggregated_lm_gap_features.data.values()
            )
            if value is not None
        )
    )
    converted_aggregated_lm_gap_features = AggregatedFeature(
        name=aggregated_lm_gap_features.name,
        data=dict(counter),
        total=counter.total(),
        statistics=aggregated_lm_gap_features.statistics,
        draw_option=aggregated_lm_gap_features.draw_option,
    )
    converted_aggregated_lm_gap_features.sort_data()
    return converted_aggregated_lm_gap_features


def save_aggregated_lm_gap_features(save_path: Path, aggregated_lm_gap_features_list: List[AggregatedFeature]) -> None:
    for aggregated_lm_gap_features in aggregated_lm_gap_features_list:
        if aggregated_lm_gap_features.name in {"masked_token_frequency"}:
            save_file_path = save_path / f"aggregation-{aggregated_lm_gap_features.name}-raw.json"
            save_json_data(aggregated_lm_gap_features.dict(), save_file_path)
            aggregated_lm_gap_features = convert_token_frequency_to_group_frequency(
                aggregated_lm_gap_features, (3, 10, 50, 100)
            )

        save_file_path = save_path / f"aggregation-{aggregated_lm_gap_features.name}.json"
        save_json_data(aggregated_lm_gap_features.dict(), save_file_path)
