#!/usr/bin/env bash

# This script will generate aggregation with plots for LM-GAP features.

set -eu

function log() {
    date_now=$(date '+%FT%T')
    echo >&2 "[${date_now}] $1"
}

DATA_PATH="scripts/results/glue/filtered_by_features/aggregation"

poetry run python -m phd_model_evaluations.cli.feature.lm_gap.aggregate_lm_gap_features \
    --input_file_path challenges/glue-lm-gap/test-A/in.tsv.xz \
    --expected_file_path challenges/glue-lm-gap/test-A/expected.tsv.xz \
    --save_path "${DATA_PATH}" \
    --skip_cache \
    --overwrite

# Generate all plots for each aggregation separately
while IFS= read -r -d '' aggregation_file_path; do
    log "[LOG] Processing aggregation for file: ${aggregation_file_path}"
    dir_path=$(dirname "${aggregation_file_path}")
    poetry run python -m phd_model_evaluations.cli.draw.draw_lm_gap_features_plot \
        --aggregated_features_path "${aggregation_file_path}" \
        --save_path "${dir_path}/plots" \
        --overwrite
done < <(find "${DATA_PATH}" -maxdepth 1 -mindepth 1 -type f -name 'aggregation-*.json' ! -name "*-raw.json" -print0 | sort --buffer-size=1G --zero-terminated)
