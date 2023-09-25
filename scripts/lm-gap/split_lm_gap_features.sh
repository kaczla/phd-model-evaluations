#!/usr/bin/env bash

# This script will split LM-GAP split by selected features.

set -eu

LATEX_TABLE_TYPE='LONGTABLE'

DATA_SET_DATA_PATH='scripts/results/glue'

function log() {
    date_now=$(date '+%FT%T')
    echo "[${date_now}] $1" | tee -a log-generating-data.log
}

function compute_correlations_results {
    if [ "$#" -lt 3 ]; then
        log '[ERROR] Missing parameters in compute_correlations_results function'
        exit 1
    fi
    results_dir="$1"
    split_name="$2"
    y_label="$3"
    extra_args=()
    file_id="${split_name}_${y_label}"

    # Get architecture type
    if [ "$#" -ge 4 ]; then
        architecture_type="$4"
        if [ "${architecture_type}" == 'encoder' ]; then
            file_id="${file_id}_encoder"
            extra_args=('--only_encoder')
        elif [ "${architecture_type}" == 'decoder' ]; then
            file_id="${file_id}_decoder"
            extra_args=('--only_decoder')
        elif [ "${architecture_type}" == 'encoder-decoder' ]; then
            file_id="${file_id}_encoder_decoder"
            extra_args=('--only_encoder_decoder')
        elif [ "${architecture_type}" != '' ]; then
            log "[ERROR] Unknown architecture type: '${architecture_type}'"
            exit 1
        fi
    fi

    previous_correlations_file_path="${DATA_SET_DATA_PATH}/finetuned_model/correlations/correlations_${file_id}.json"
    extra_args+=('--previous_correlations_file_path')
    extra_args+=("${previous_correlations_file_path}")

    table_caption=${TABLE_CAPTION:="Correlations results on ${split_name} set (${y_label})"}
    table_label=${TABLE_LABEL:="table:glue_correlations_${file_id}"}

    results_file="${results_dir}/model_${split_name}_results.json"
    save_file="${results_dir}/correlations/correlations_${file_id}.json"
    log "[INFO] Computing correlations for:"
    log "      * results file: ${results_file}"
    log "      * extra args  : ${extra_args[*]}"
    log "      into          : ${save_file}"
    poetry run python -m phd_model_evaluations.cli.statistics.compute_correlations \
        --file_path "${results_file}" \
        --save_path "${save_file}" \
        --y_labels "${y_label}" \
        "${extra_args[@]}" \
        --save_table_data

    table_file="${results_dir}/correlations/table-correlations_${file_id}.json"
    save_tex_file="${results_dir}/correlations/table-correlations_${file_id}.tex"
    log "[INFO] Generating Latex table for:"
    log "      * table file: ${table_file}"
    log "      * caption   : ${table_caption}"
    log "      * label     : ${table_label}"
    log "      into        : ${save_tex_file}"
    poetry run python -m phd_model_evaluations.cli.output.generate_latex_table \
        --table_path "${table_file}" \
        --save_path "${save_tex_file}" \
        --label "${table_label}" \
        --caption "${table_caption}" \
        --table_type "${LATEX_TABLE_TYPE:="TABLE"}" \
        --rotate_header_table True \
        --mapping_parbox_size '{"1": 4.5, "2": 4.5, "3": 4.5, "4": 4.5, "5": 4.5, "6": 4.5}'
}

function generate_aggregated_results_with_lm_gap_and_loss {
    if [ "$#" -lt 4 ]; then
        log '[ERROR] Missing parameters in generate_aggregated_results_with_lm_gap_and_loss function'
        exit 1
    fi
    results_dir="$1"
    lm_gap_dir="$2"
    loss_dir="$3"
    split_name="$4"

    dataset_name=${DATASET_NAME:=''}
    results_file="${results_dir}/raw_model_${split_name}_results.json"
    generate_latex_table='False'
    lm_gap_score_key=${LM_GAP_SCORE_KEY:='PerplexityHashed'}
    table_caption=${TABLE_CAPTION:="GLUE Benchmark scores on ${split_name} set"}
    table_label=${TABLE_LABEL:="table:glue_score_${split_name}"}

    if [ "${lm_gap_score_key}" != 'PerplexityHashed' ]; then
        results_dir="${results_dir}/LM-GAP/${lm_gap_score_key}"
        mkdir -p "${results_dir}"
        generate_latex_table="False"
    fi

    if [ "${dataset_name}" == '' ]; then
        save_file="${results_dir}/model_${split_name}_results.json"
        table_file="${results_dir}/table-model_${split_name}_results.json"
        save_tex_file="${results_dir}/table-model_${split_name}_results.tex"
    else
        results_dir="${results_dir}/datasets"
        generate_latex_table="False"
        mkdir -p "${results_dir}"
        save_file="${results_dir}/model_${split_name}_${dataset_name}_results.json"
        table_file="${results_dir}/table-model_${split_name}_${dataset_name}_results.json"
        save_tex_file="${results_dir}/table-model_${split_name}_${dataset_name}_results.tex"
    fi

    lm_gap_file="${lm_gap_dir}/${split_name}.json"
    loss_file="${loss_dir}/${split_name}.json"
    log "[INFO] Generating aggregated results for:"
    log "      * results file: ${results_file}"
    log "      * LM-GAP file : ${lm_gap_file}"
    log "      * LM-GAP key  : ${lm_gap_score_key}"
    log "      * loss file   : ${loss_file}"
    log "      into          : ${save_file}"
    poetry run python -m phd_model_evaluations.cli.output.generate_aggregated_results \
        --results_file_path "${results_file}" \
        --lm_gap_file_path "${lm_gap_file}" \
        --lm_gap_score_key_name "${lm_gap_score_key}" \
        --loss_file_path "${loss_file}" \
        --save_path "${save_file}" \
        --save_table_data "${generate_latex_table}" \
        --add_average_score True \
        --return_empty_date True
}

function split_lm_gap_features() {
    if [ "$#" -lt 1 ]; then
        log '[ERROR] Missing arguments: target data-set name'
        exit 1
    fi
    target_set_name="$1"
    challenge_path=${LM_GAP_CHALLENGE_PATH:='challenges/glue-lm-gap'}
    challenge_set_name=${LM_GAP_CHALLENGE_SET_NAME:='test-A'}
    log "[INFO] Splitting LM-GAP features for ${target_set_name} ..."

    if [ ! -d "${challenge_path}" ]; then
        log "[ERROR] Missing LM-GAP challenge path: ${challenge_path}"
        exit 2
    fi

    set_path="${challenge_path}/${challenge_set_name}"
    save_challenge_path="${challenge_path}/features"
    target_set_path="features/${target_set_name}"
    mkdir -p "${save_challenge_path}"
    save_challenge_path="${save_challenge_path}/${target_set_name}"
    checker_file_path=${LM_GAP_FEATURE_CHECKER_PATH:=''}
    if [ -f "${checker_file_path}" ]; then
        extra_args=('--checkers_file_path' "${checker_file_path}")
    fi

    log "[INFO] Splitting LM-GAP features:"
    log "      * set path  : ${set_path}"
    log "      * extra args: ${extra_args[*]}"
    log "      into        : ${save_challenge_path}"
    poetry run python -m phd_model_evaluations.cli.feature.lm_gap.split_lm_gap_by_features \
        --set_path "${set_path}" \
        --save_path "${save_challenge_path}" \
        "${extra_args[@]}"

    log "[INFO] Evaluating in ${challenge_path}"
    cd "${challenge_path}"
    bash run_parallel_geval.sh "${target_set_path}"
    cp "tmp_evaluation-${target_set_name}/results.tsv" "results-${target_set_name}.tsv"
    cd -

    DATA_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/${target_set_path}"
    DATA_MODEL_PATH="${DATA_PATH}/finetuned_model"
    DATA_LM_GAP_PATH="${DATA_PATH}/glue-lm-gap"
    DATA_LOSS_PATH="${DATA_PATH}/loss"

    log '[INFO] Preparing model results ...'
    mkdir -p "${DATA_PATH}"
    mkdir -p "${DATA_MODEL_PATH}" "${DATA_LM_GAP_PATH}" "${DATA_LOSS_PATH}"
    cp -v "${DATA_SET_DATA_PATH}/finetuned_model/raw_model_validation_results.json" "${DATA_MODEL_PATH}"
    cp -v "${DATA_SET_DATA_PATH}/loss/validation.json" "${DATA_LOSS_PATH}"
    cp -v "${challenge_path}/results-${target_set_name}.tsv" "${DATA_LM_GAP_PATH}/validation.tsv"
    cp -v "${challenge_path}/${target_set_path}/"{lm_gap_feature_checkers.json,selected_lines.json} "${DATA_PATH}"

    poetry run python -m phd_model_evaluations.cli.output.convert_tsv_to_json \
        --input_file "${DATA_LM_GAP_PATH}/validation.tsv" \
        --output_file "${DATA_LM_GAP_PATH}/validation.json" \
        --sort_key_name 'model_name'

    generate_aggregated_results_with_lm_gap_and_loss "${DATA_MODEL_PATH}" "${DATA_LM_GAP_PATH}" "${DATA_LOSS_PATH}" "validation"

    log "[INFO] Computing correlations ..."
    mkdir -p "${DATA_MODEL_PATH}/correlations"
    TABLE_CAPTION="Korelacja wyników neuronowych modeli języka pomiędzy wynikiem zadania zgadywania zamaskowanego słowa oraz wynikami na zbiorze zadań GLUE Benchmark na ograniczonym zbiorze."
    TABLE_LABEL="table:glue_correlations_validation_lm_gap_feature_${target_set_name}"
    compute_correlations_results "${DATA_MODEL_PATH}" "validation" "LM-GAP"
    TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o koder architektury Transformer pomiędzy wynikiem zadania zgadywania zamaskowanego słowa oraz wynikami na zbiorze zadań GLUE Benchmark na ograniczonym zbiorze."
    TABLE_LABEL="table:glue_correlations_validation_lm_gap_feature_${target_set_name}_encoder"
    compute_correlations_results "${DATA_MODEL_PATH}" "validation" "LM-GAP" "encoder"
    TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o dekoder architektury Transformer pomiędzy wynikiem zadania zgadywania zamaskowanego słowa oraz wynikami na zbiorze zadań GLUE Benchmark na ograniczonym zbiorze."
    TABLE_LABEL="table:glue_correlations_validation_lm_gap_feature_${target_set_name}_decoder"
    compute_correlations_results "${DATA_MODEL_PATH}" "validation" "LM-GAP" "decoder"
    TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o koder-dekoder architektury Transformer pomiędzy wynikiem zadania zgadywania zamaskowanego słowa oraz wynikami na zbiorze zadań GLUE Benchmark na ograniczonym zbiorze."
    TABLE_LABEL="table:glue_correlations_validation_lm_gap_feature_${target_set_name}_encoder_decoder"
    compute_correlations_results "${DATA_MODEL_PATH}" "validation" "LM-GAP" "encoder-decoder"

    log '[INFO] Done'
}

function save_aggregated_lm_gap_results() {
    log "[INFO] Saving aggregated LM-GAP results ..."

    use_raw_name=${USE_RAW_NAME_IN_HEADER:=0}
    counter=0

    # Get all files automatically
    file_paths=()
    file_names=()
    while IFS= read -r -d '' split_path; do
        validation_file_path="${split_path}/glue-lm-gap/validation.json"
        if [ ! -f "${validation_file_path}" ]; then
            log "[ERROR] Missing LM-GAP results in ${split_path}"
            continue
        fi

        split_name=$(basename "${split_path}")
        log "[INFO] Using LM-GAP results for: ${split_name}"
        counter=$((counter + 1))

        file_paths+=("${validation_file_path}")
        if [ "${use_raw_name}" -gt 0 ]; then
            file_names+=("${split_name}")
        else
            file_names+=("${counter}")
        fi
    done < <(find "${DATA_SET_DATA_PATH}/filtered_by_features/features" -maxdepth 1 -mindepth 1 -type d -print0 | sort --buffer-size=1G --zero-terminated)

    # Aggregate results
    poetry run python -m phd_model_evaluations.cli.output.lm_gap.generate_aggregated_lm_gap_results \
        --source_file_path \
        "${DATA_SET_DATA_PATH}/glue-lm-gap/validation.json" \
        --other_file_paths "${file_paths[@]}" \
        --name_other_files "${file_names[@]}" \
        --generate_in_groups \
        --save_path "${DATA_SET_DATA_PATH}/filtered_by_features/merged_glue-lm-gap/validation.json" \
        --save_table_data

    # Generate LaTeX table
    counter=0
    while IFS= read -r -d '' table_file; do
        # Whole table
        table_caption="Porównanie wyników na zadaniu zgadywania zamaskowanego słowa dla poszczególnych podzbiorów na zbiorze zadań GLUE Benchmark - część $((counter + 1))."
        table_label="table:glue_lm_gap_feature_validation_comparing_${counter}"
        save_tex_file="${DATA_SET_DATA_PATH}/filtered_by_features/merged_glue-lm-gap/lm_gap_features_merged_results-group_${counter}.tex"
        log "[INFO] Generating Latex table for:"
        log "      * table file: ${table_file}"
        log "      * caption   : ${table_caption}"
        log "      * label     : ${table_label}"
        log "      into        : ${save_tex_file}"
        poetry run python -m phd_model_evaluations.cli.output.generate_latex_table \
            --table_path "${table_file}" \
            --save_path "${save_tex_file}" \
            --label "${table_label}" \
            --caption "${table_caption}" \
            --mapping_label '{"AVG": "Średnia", "gap_with_punctuation_1": "A1", "gap_with_punctuation_2": "A2", "gap_with_punctuation_3": "A3", "gap_with_punctuation_4": "A4", "is_number": "B", "masked_token_frequency_1": "C1", "masked_token_frequency_2": "C2", "masked_token_frequency_3": "C3", "masked_token_frequency_4": "C4", "masked_token_length_1": "D1", "masked_token_length_2": "D2", "masked_token_length_3": "D3", "masked_token_length_4": "D4", "left_context_length_1": "E1", "left_context_length_2": "E2", "left_context_length_3": "E3", "left_context_length_4": "E4", "right_context_length_1": "F1", "right_context_length_2": "F2", "right_context_length_3": "F3", "right_context_length_4": "F4", "text_length_1": "G1", "text_length_2": "G2", "text_length_3": "G3", "text_length_4": "G4", "text_length_5": "G5"}' \
            --table_type "${LATEX_TABLE_TYPE:="TABLE"}"

        # Part table - selected rows
        table_caption="Porównanie wybranych wyników na zadaniu zgadywania zamaskowanego słowa dla poszczególnych podzbiorów na zbiorze zadań GLUE Benchmark - część $((counter + 1))."
        table_label="table:glue_lm_gap_feature_validation_comparing_${counter}_part"
        save_tex_file="${DATA_SET_DATA_PATH}/filtered_by_features/merged_glue-lm-gap/lm_gap_features_merged_results-group_${counter}_part.tex"
        log "[INFO] Generating Latex table for:"
        log "      * table file: ${table_file}"
        log "      * caption   : ${table_caption}"
        log "      * label     : ${table_label}"
        log "      into        : ${save_tex_file}"
        poetry run python -m phd_model_evaluations.cli.output.generate_latex_table \
            --table_path "${table_file}" \
            --save_path "${save_tex_file}" \
            --label "${table_label}" \
            --caption "${table_caption}" \
            --mapping_label '{"AVG": "Średnia", "gap_with_punctuation_1": "A1", "gap_with_punctuation_2": "A2", "gap_with_punctuation_3": "A3", "gap_with_punctuation_4": "A4", "is_number": "B", "masked_token_frequency_1": "C1", "masked_token_frequency_2": "C2", "masked_token_frequency_3": "C3", "masked_token_frequency_4": "C4", "masked_token_length_1": "D1", "masked_token_length_2": "D2", "masked_token_length_3": "D3", "masked_token_length_4": "D4", "left_context_length_1": "E1", "left_context_length_2": "E2", "left_context_length_3": "E3", "left_context_length_4": "E4", "right_context_length_1": "F1", "right_context_length_2": "F2", "right_context_length_3": "F3", "right_context_length_4": "F4", "text_length_1": "G1", "text_length_2": "G2", "text_length_3": "G3", "text_length_4": "G4", "text_length_5": "G5"}' \
            --selected_row_names 'BERT-base-cased' 'BERT-base-uncased' 'FLAN-T5-base' 'GPT-2-base' 'OPT-125M' 'Pythia-160M' 'RoBERTa-base' 'T5-base' 'mT5-base' 'Średnia' \
            --table_type "${LATEX_TABLE_TYPE:="TABLE"}"

        counter=$((counter + 1))
    done < <(find "${DATA_SET_DATA_PATH}/filtered_by_features/merged_glue-lm-gap/" -maxdepth 1 -mindepth 1 -type f -name "table-*.json" -print0 | sort --buffer-size=1G --zero-terminated)
}

# Dumping example LM-GAP feature checkers
DUMP_EXAMPLE_CHECKERS=0
if [ "${DUMP_EXAMPLE_CHECKERS}" -gt 0 ]; then
    log "[INFO] Dumping example of LM-GAP feature checkers ..."
    poetry run python -m phd_model_evaluations.cli.feature.lm_gap.dump_example_lm_gap_feature_checkers \
        --save_path "${DATA_SET_DATA_PATH}/filtered_by_features/features/example_lm_gap_feature_checkers.json"
fi

# Generate data for each LM-GAP feature
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/is_number/lm_gap_feature_checkers.json" split_lm_gap_features "is_number"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/gap_with_punctuation_1/lm_gap_feature_checkers.json" split_lm_gap_features "gap_with_punctuation_1"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/gap_with_punctuation_2/lm_gap_feature_checkers.json" split_lm_gap_features "gap_with_punctuation_2"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/gap_with_punctuation_3/lm_gap_feature_checkers.json" split_lm_gap_features "gap_with_punctuation_3"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/gap_with_punctuation_4/lm_gap_feature_checkers.json" split_lm_gap_features "gap_with_punctuation_4"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/masked_token_frequency_1/lm_gap_feature_checkers.json" split_lm_gap_features "masked_token_frequency_1"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/masked_token_frequency_2/lm_gap_feature_checkers.json" split_lm_gap_features "masked_token_frequency_2"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/masked_token_frequency_3/lm_gap_feature_checkers.json" split_lm_gap_features "masked_token_frequency_3"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/masked_token_frequency_4/lm_gap_feature_checkers.json" split_lm_gap_features "masked_token_frequency_4"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/left_context_length_1/lm_gap_feature_checkers.json" split_lm_gap_features "left_context_length_1"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/left_context_length_2/lm_gap_feature_checkers.json" split_lm_gap_features "left_context_length_2"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/left_context_length_3/lm_gap_feature_checkers.json" split_lm_gap_features "left_context_length_3"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/left_context_length_4/lm_gap_feature_checkers.json" split_lm_gap_features "left_context_length_4"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/right_context_length_1/lm_gap_feature_checkers.json" split_lm_gap_features "right_context_length_1"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/right_context_length_2/lm_gap_feature_checkers.json" split_lm_gap_features "right_context_length_2"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/right_context_length_3/lm_gap_feature_checkers.json" split_lm_gap_features "right_context_length_3"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/right_context_length_4/lm_gap_feature_checkers.json" split_lm_gap_features "right_context_length_4"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/text_length_1/lm_gap_feature_checkers.json" split_lm_gap_features "text_length_1"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/text_length_2/lm_gap_feature_checkers.json" split_lm_gap_features "text_length_2"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/text_length_3/lm_gap_feature_checkers.json" split_lm_gap_features "text_length_3"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/text_length_4/lm_gap_feature_checkers.json" split_lm_gap_features "text_length_4"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/text_length_5/lm_gap_feature_checkers.json" split_lm_gap_features "text_length_5"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/masked_token_length_1/lm_gap_feature_checkers.json" split_lm_gap_features "masked_token_length_1"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/masked_token_length_2/lm_gap_feature_checkers.json" split_lm_gap_features "masked_token_length_2"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/masked_token_length_3/lm_gap_feature_checkers.json" split_lm_gap_features "masked_token_length_3"
LM_GAP_FEATURE_CHECKER_PATH="${DATA_SET_DATA_PATH}/filtered_by_features/features/masked_token_length_4/lm_gap_feature_checkers.json" split_lm_gap_features "masked_token_length_4"

# Saving aggregated LM_GAP results
USE_RAW_NAME_IN_HEADER=1 save_aggregated_lm_gap_results
