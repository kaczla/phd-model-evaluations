#!/usr/bin/env bash

# This script copy generated Latex tables into my PhD.

set -eu

function log() {
    date_now=$(date '+%FT%T')
    echo >&2 "[${date_now}] $1"
}

SAVE_PATH="phd-kaczmarek/tables"
if [ ! -d "${SAVE_PATH}" ]; then
    log "[ERROR] Save path: ${SAVE_PATH} does not exist!"
    exit 2
fi

log "[INFO] Coping Latex tables..."

RESULTS_PATH='scripts/results'

# Model parameters
cp -v "${RESULTS_PATH}/table-model_parameters.tex" "${SAVE_PATH}/model_parameters.tex"
cp -v "${RESULTS_PATH}/table-model_information.tex" "${SAVE_PATH}/model_information.tex"

# GLUE Benchmark
DATA_PATH="${RESULTS_PATH}/glue"

# Self fine-tuned models
cp -v "${DATA_PATH}/finetuned_model/table-model_validation_results.tex" "${SAVE_PATH}/glue_fine_tuned_validation_score.tex"
cp -v "${DATA_PATH}/finetuned_model/correlations/table-correlations_validation_loss.tex" "${SAVE_PATH}/glue_fine_tuned_validation_correlations_loss.tex"
cp -v "${DATA_PATH}/finetuned_model/correlations/table-correlations_validation_LM-GAP.tex" "${SAVE_PATH}/glue_fine_tuned_validation_correlations_LM_GAP.tex"
for architecture_name in 'encoder' 'decoder' 'encoder_decoder'; do
    cp -v "${DATA_PATH}/finetuned_model/correlations/table-correlations_validation_loss_${architecture_name}.tex" "${SAVE_PATH}/glue_fine_tuned_validation_correlations_loss_${architecture_name}.tex"
    cp -v "${DATA_PATH}/finetuned_model/correlations/table-correlations_validation_LM-GAP_${architecture_name}.tex" "${SAVE_PATH}/glue_fine_tuned_validation_correlations_LM_GAP_${architecture_name}.tex"
done

# Official models
cp -v "${DATA_PATH}/official_model/table-model_validation_results.tex" "${SAVE_PATH}/glue_official_validation_score.tex"
cp -v "${DATA_PATH}/official_model/table-model_test_results.tex" "${SAVE_PATH}/glue_official_test_score.tex"

# LM-GAP features
SAVE_LM_GAP_FEATURES_PATH="${SAVE_PATH}/lm_gap_features"
SAVE_LM_GAP_FEATURES_PATH_IMAGES="${SAVE_PATH_IMAGES}/lm_gap_features"
mkdir -p "${SAVE_LM_GAP_FEATURES_PATH}" "${SAVE_LM_GAP_FEATURES_PATH_IMAGES}"
# Merged LM-GAP results
cp -v "${RESULTS_PATH}/glue/filtered_by_features/merged_glue-lm-gap/lm_gap_features_merged_results*.tex" "${SAVE_PATH}/lm_gap_features"
# Diagrams
while IFS= read -r -d '' plot_path; do
    plot_name=$(basename "${plot_path}")
    cp -v "${DATA_PATH}/filtered_by_features/aggregation/plots/${plot_name}" "${SAVE_LM_GAP_FEATURES_PATH_IMAGES}/lm_gap_features_${plot_name}"
done < <(find "${DATA_PATH}/filtered_by_features/aggregation/plots" -maxdepth 1 -mindepth 1 -type f -print0 | sort --buffer-size=1G --zero-terminated)
# Correlation tables
while IFS= read -r -d '' split_path; do
    split_name=$(basename "${split_path}")
    cp -v "${DATA_PATH}/filtered_by_features/features/${split_name}/finetuned_model/correlations/table-correlations_validation_LM-GAP.tex" "${SAVE_LM_GAP_FEATURES_PATH}/lm_gap_features_correlations_${split_name}.tex"
    cp -v "${DATA_PATH}/filtered_by_features/features/${split_name}/finetuned_model/correlations/table-correlations_validation_LM-GAP_decoder.tex" "${SAVE_LM_GAP_FEATURES_PATH}/lm_gap_features_correlations_${split_name}_decoder.tex"
    cp -v "${DATA_PATH}/filtered_by_features/features/${split_name}/finetuned_model/correlations/table-correlations_validation_LM-GAP_encoder.tex" "${SAVE_LM_GAP_FEATURES_PATH}/lm_gap_features_correlations_${split_name}_encoder.tex"
    cp -v "${DATA_PATH}/filtered_by_features/features/${split_name}/finetuned_model/correlations/table-correlations_validation_LM-GAP_encoder_decoder.tex" "${SAVE_LM_GAP_FEATURES_PATH}/lm_gap_features_correlations_${split_name}_encoder_decoder.tex"
done < <(find "${DATA_PATH}/filtered_by_features/features" -maxdepth 1 -mindepth 1 -type d -print0 | sort --buffer-size=1G --zero-terminated)

log "[INFO] Latex tables copied"
