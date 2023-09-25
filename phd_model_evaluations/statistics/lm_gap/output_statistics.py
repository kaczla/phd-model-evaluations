import logging
from collections import Counter
from math import isclose
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from phd_model_evaluations.data.lm_gap.lm_gap_output_statistics import LMGapOutputStatistics
from phd_model_evaluations.data.prediction_base import PredictionBase
from phd_model_evaluations.statistics.generate_positions_statistics import get_positions_statistics
from phd_model_evaluations.statistics.generate_probability_statistics import generate_score_statistics
from phd_model_evaluations.utils.common_utils import get_open_fn

LOGGER = logging.getLogger(__name__)


def generate_statists(all_predictions: List[List[PredictionBase]], expected_tokens: List[str]) -> LMGapOutputStatistics:
    unk_probabilities = []
    first_token_probabilities = []
    token_match_positions: CounterType = Counter()
    total_not_equal_sum = 0
    total_match_tokens = 0
    for i_prediction, (predictions, expected_token) in enumerate(
        tqdm(zip(all_predictions, expected_tokens, strict=True), desc="Collecting statistics"), start=1
    ):
        found_unk = False
        match_token = False
        tokens = set()
        total_score = 0.0
        for i, prediction in enumerate(predictions, start=1):
            token = prediction.token
            score = prediction.score
            tokens.add(token)
            total_score += score

            if not token:
                unk_probabilities.append(score)
                found_unk = True
            elif token == expected_token:
                token_match_positions.update([i])
                match_token = True

            if i == 1:
                first_token_probabilities.append(score)

        if len(tokens) != len(predictions):
            LOGGER.error(
                f"[{i_prediction}] Found duplicated tokens, found {len(tokens)} unique tokens,"
                f" but are {len(predictions)} tokens"
            )
        if not found_unk:
            LOGGER.warning(f"[{i_prediction}] Not found probability for UNK!")

        total_not_equal_sum += int(not isclose(total_score, 1.0, abs_tol=1e-04))
        total_match_tokens += int(match_token)

    total_lines = len(all_predictions)
    return LMGapOutputStatistics(
        total_lines=total_lines,
        total_probabilities_not_summed_to_one=total_not_equal_sum,
        total_token_match=total_match_tokens,
        total_token_not_match=total_lines - total_match_tokens,
        unk_statistics=generate_score_statistics(np.array(unk_probabilities)),
        first_token_statistics=generate_score_statistics(np.array(first_token_probabilities)),
        token_match_positions=get_positions_statistics(
            token_match_positions, [1, 2, 3, 5, 10, 15, 25, 50, 100, 150, 200, 250, 500, 1000]
        ),
    )


def parse_predictions(line: str, number_line: int) -> List[PredictionBase]:
    line = line.strip()
    predictions = line.split(" ")

    parsed_predictions = []
    for prediction in predictions:
        if ":" not in prediction:
            raw_prediction = repr(prediction)
            LOGGER.error(f"[{number_line}] Invalid prediction: {raw_prediction}")
            continue

        token, score_str = prediction.rsplit(":", 1)
        try:
            score = float(score_str)
        except ValueError:
            raw_prediction = repr(prediction)
            LOGGER.error(f"[{number_line}] Cannot parse score in prediction: {raw_prediction}")
            continue

        parsed_predictions.append(PredictionBase(token=token, score=score))

    return parsed_predictions


def parse_expected_tokens(file_path: Path) -> List[str]:
    LOGGER.info(f"Loading expected tokens from: {file_path}")
    open_fn = get_open_fn(file_path.name)

    with open_fn(file_path, "rt") as f_read:
        expected_tokens = [line.strip() for line in tqdm(f_read, desc="Reading expected tokens")]
    LOGGER.info(f"Loaded {len(expected_tokens)} expected tokens")

    return expected_tokens


def generate_single_lm_gap_output_statists(file_path: Path, expected_tokens: List[str]) -> LMGapOutputStatistics:
    LOGGER.info(f"Loading predictions from: {file_path}")
    open_fn = get_open_fn(file_path.name)
    with open_fn(file_path, "rt") as f_read:
        all_predictions = []
        for number_line, line in enumerate(tqdm(f_read, desc="Reading predictions"), start=1):
            parsed_predictions = parse_predictions(line, number_line)
            all_predictions.append(parsed_predictions)

    LOGGER.info(f"Loaded {len(all_predictions)} prediction lines")
    return generate_statists(all_predictions, expected_tokens)


def generate_lm_gap_output_statists(directory_path: Path) -> Dict[str, LMGapOutputStatistics]:
    LOGGER.info(f"Processing directory: {directory_path}")

    expected_file_paths, output_file_paths = [], []
    for sub_path in directory_path.iterdir():
        if not sub_path.is_file():
            continue

        if sub_path.name.startswith("expected.tsv"):
            expected_file_paths.append(sub_path)

        if sub_path.name.startswith("out"):
            output_file_paths.append(sub_path)
    LOGGER.info(f"Found {len(expected_file_paths)} expected files and {len(output_file_paths)} output files")

    all_statistics = {}
    expected_tokens = parse_expected_tokens(sorted(expected_file_paths)[0])
    for output_file_path in sorted(output_file_paths):
        statistics = generate_single_lm_gap_output_statists(output_file_path, expected_tokens)
        LOGGER.info(f"Statistics: {statistics.dict()}")
        all_statistics[output_file_path.name] = statistics

    return all_statistics
