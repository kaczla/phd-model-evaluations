#!/usr/bin/env bash

set -eu

function log() {
    date_now=$(date '+%FT%T')
    echo "[${date_now}] $1" | tee -a log-generating-data.log
}

export TOKENIZERS_PARALLELISM='true'
export CACHE_DIRECTORY='.cache'
export HF_HOME="${CACHE_DIRECTORY}"
export HF_DATASETS_CACHE="${CACHE_DIRECTORY}/datasets"
export TRANSFORMERS_CACHE="${CACHE_DIRECTORY}/transformers"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

if [ ! -d "${CACHE_DIRECTORY}" ]; then
    log '[ERROR] Not found ".cache" directory in current directory!'
    exit 1
fi

function generate_model_parameters {
    if [ "$#" -lt 1 ]; then
        log '[ERROR] Missing parameters in generate_model_parameters function'
        exit 1
    fi
    save_dir="$1"

    save_file_name_without_extension='model_parameters'
    save_file_name="${save_file_name_without_extension}.json"
    save_file="${save_dir}/${save_file_name}"
    log "[INFO] Generating models parameters:"
    log "      into          : ${save_file}"
    poetry run python -m phd_model_evaluations.cli.output.generate_model_parameters \
        --save_path "${save_file}" \
        --save_table_data True

    table_file="${save_dir}/table-${save_file_name}"
    save_tex_file="${save_dir}/table-${save_file_name_without_extension}.tex"
    log "[INFO] Generating Latex table for:"
    log "      * table file: ${table_file}"
    log "      into        : ${save_tex_file}"
    poetry run python -m phd_model_evaluations.cli.output.generate_latex_table \
        --table_path "${table_file}" \
        --save_path "${save_tex_file}" \
        --label 'table:model_parameters' \
        --caption 'Lista publicznie dostępnych neuronowych modeli języka oparte na architekturze Transformer. Oznaczenie \textbf{K}, \textbf{D} oznacza odpowiednio parametry dla koder oraz dekoder architektury Transformer.' \
        --table_type "${LATEX_TABLE_TYPE:="TABLE"}" \
        --rotate_header_table True \
        --mapping_parbox_size '{"6": 3, "7": 3, "8": 3}'
}

function generate_model_information {
    if [ "$#" -lt 1 ]; then
        log '[ERROR] Missing parameters in generate_model_information function'
        exit 1
    fi
    save_dir="$1"

    save_file_name_without_extension='model_information'
    save_file_name="${save_file_name_without_extension}.json"
    save_file="${save_dir}/${save_file_name}"
    log "[INFO] Generating models information:"
    log "      into          : ${save_file}"
    poetry run python -m phd_model_evaluations.cli.output.generate_model_information \
        --save_path "${save_file}" \
        --save_table_data True

    table_file="${save_dir}/table-${save_file_name}"
    save_tex_file="${save_dir}/table-${save_file_name_without_extension}.tex"
    log "[INFO] Generating Latex table for:"
    log "      * table file: ${table_file}"
    log "      into        : ${save_tex_file}"
    poetry run python -m phd_model_evaluations.cli.output.generate_latex_table \
        --table_path "${table_file}" \
        --save_path "${save_tex_file}" \
        --label 'table:model_information' \
        --caption 'Lista wykorzystanych neuronowych modeli języka opartych na architekturze Transformer.' \
        --table_type "${LATEX_TABLE_TYPE:="TABLE"}"
}

function generate_lm_gap_data {
    if [ "$#" -lt 3 ]; then
        log '[ERROR] Missing parameters in generate_lm_gap_data function'
        exit 1
    fi
    lm_gap_challenge_dir="$1"
    set_name="$2"
    save_dir="$3"

    # Get save file name
    if [ "${set_name}" == 'dev-0' ]; then
        save_file_name='part-train.json'
    elif [ "${set_name}" == 'test-A' ]; then
        save_file_name='validation.json'
    else
        log "[ERROR] Cannot detect save file name for set name: ${set_name}"
        exit 2
    fi

    lm_gap_file_name="results-${set_name}.tsv"
    lm_gap_file="${lm_gap_challenge_dir}/${lm_gap_file_name}"
    log "[INFO] Computing LM-GAP scores with geval:"
    log "      into          : ${lm_gap_file}"
    cd "${lm_gap_challenge_dir}"
    geval -t "${set_name}" >"${lm_gap_file_name}" || true
    cd - &>/dev/null

    save_file="${save_dir}/${save_file_name}"
    log "[INFO] Converting LM-GAP data to JSON:"
    log "      * LM-GAP scores: ${lm_gap_file}"
    log "      into           : ${save_file}"
    poetry run python -m phd_model_evaluations.cli.output.convert_tsv_to_json \
        --input_file "${lm_gap_file}" \
        --output_file "${save_file}" \
        --sort_key_name 'model_name'
}

function generate_models_results {
    if [ "$#" -lt 3 ]; then
        log '[ERROR] Missing parameters in generate_models_results function'
        exit 1
    fi
    saved_models_dir="$1"
    results_dir="$2"
    split_name="$3"

    save_file="${results_dir}/raw_model_${split_name}_results.json"
    log "[INFO] Generating models results for:"
    log "      * models directory: ${saved_models_dir}"
    log "      into          : ${save_file}"
    poetry run python -m phd_model_evaluations.cli.output.generate_models_results \
        --models_dir_path "${saved_models_dir}" \
        --save_path "${save_file}" \
        --return_empty_date True
}

function generate_aggregated_results {
    if [ "$#" -lt 2 ]; then
        log '[ERROR] Missing parameters in generate_aggregated_results function'
        exit 1
    fi
    results_dir="$1"
    split_name="$2"
    table_caption=${TABLE_CAPTION:="GLUE Benchmark scores on ${split_name} set"}
    table_label=${TABLE_LABEL:="table:glue_score_${split_name}"}

    results_file="${results_dir}/raw_model_${split_name}_results.json"
    save_file="${results_dir}/model_${split_name}_results.json"
    log "[INFO] Generating aggregated results for:"
    log "      * results file: ${results_file}"
    log "      into          : ${save_file}"
    poetry run python -m phd_model_evaluations.cli.output.generate_aggregated_results \
        --results_file_path "${results_file}" \
        --save_path "${save_file}" \
        --save_table_data True \
        --add_average_score True \
        --return_empty_date True

    table_file="${results_dir}/table-model_${split_name}_results.json"
    save_tex_file="${results_dir}/table-model_${split_name}_results.tex"
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
        --rotate_header_table True
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
    generate_latex_table='True'
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

    if [ "${generate_latex_table}" == 'True' ]; then
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
            --rotate_header_table True
    fi
}

function draw_plot {
    if [ "$#" -lt 2 ]; then
        log '[ERROR] Missing parameters in draw_plot function'
        exit 1
    fi
    results_dir="$1"
    split_name="$2"
    dataset_name=${DATASET_NAME:=''}
    lm_gap_score_key=${LM_GAP_SCORE_KEY:='PerplexityHashed'}

    if [ "${lm_gap_score_key}" != 'PerplexityHashed' ]; then
        results_dir="${results_dir}/LM-GAP/${lm_gap_score_key}"
    fi

    if [ "${dataset_name}" == '' ]; then
        results_file="${results_dir}/model_${split_name}_results.json"
        save_loss_plot_path="${results_dir}/plot-loss-${split_name}.png"
        save_lm_gap_plot_path="${results_dir}/plot-lm-gap-${split_name}.png"
        y_names=('MNLI-m' 'MNLI-mm' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST-2' 'STS-B')
        extra_args=('--add_label_at_top')
    else
        results_dir="${results_dir}/datasets"
        results_file="${results_dir}/model_${split_name}_${dataset_name}_results.json"
        save_loss_plot_path="${results_dir}/plot-loss-${split_name}-${dataset_name}.png"
        save_lm_gap_plot_path="${results_dir}/plot-lm-gap-${split_name}-${dataset_name}.png"
        y_names=("${dataset_name}")
        extra_args=()
    fi

    log "[INFO] Drawing loss plot for:"
    log "      * dataset name: ${dataset_name}"
    log "      * results file: ${results_file}"
    log "      * Y labels    : ${y_names[*]}"
    log "      * LM-GAP key  : ${lm_gap_score_key}"
    log "      * extra args  : ${extra_args[*]}"
    log "      into          : ${save_loss_plot_path}"
    poetry run python -m phd_model_evaluations.cli.draw.draw_plot \
        --aggregated_results_path "${results_file}" \
        --save_path "${save_loss_plot_path}" \
        --x_name 'loss' \
        --y_names "${y_names[@]}" \
        --legend_columns 2 \
        --x_title "Loss from pre-trained model" \
        --y_title "GLUE Benchmark score" \
        --add_linear_regression \
        "${extra_args[@]}" \
        --max_x_value 5

    log "[INFO] Drawing LM-GAP plot for:"
    log "      * results file: ${results_file}"
    log "      into          : ${save_lm_gap_plot_path}"
    poetry run python -m phd_model_evaluations.cli.draw.draw_plot \
        --aggregated_results_path "${results_file}" \
        --save_path "${save_lm_gap_plot_path}" \
        --x_name 'LM-GAP' \
        --y_names "${y_names[@]}" \
        --legend_columns 2 \
        --x_title "LM-GAP score" \
        --y_title "GLUE Benchmark score" \
        --add_linear_regression \
        "${extra_args[@]}" \
        --max_x_value 2500
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
        --mapping_parbox_size '{"1": 4.5, "2": 4.5, "3": 4.5, "4": 4.5}'
}

LATEX_TABLE_TYPE='LONGTABLE'

RESULTS_PATH='scripts/results'

# Model parameters
GENERATE_MODEL_DATA=0
# Skip generating if no needed
if [ "${GENERATE_MODEL_DATA}" -gt 0 ]; then
    log "[INFO] Generating model parameters ..."
    generate_model_parameters "${RESULTS_PATH}"
    generate_model_information "${RESULTS_PATH}"
fi

# GLUE Benchmark
DATA_PATH="${RESULTS_PATH}/glue"
LM_GAP_CHALLENGE_PATH="challenges/glue-lm-gap"

# Get LM-GAP scores
GENERATE_LM_GAP_DATA=0
# Skip generating if no needed
if [ "${GENERATE_LM_GAP_DATA}" -gt 0 ]; then
    log "[INFO] Generating LM-GAP data ..."
    generate_lm_gap_data "${LM_GAP_CHALLENGE_PATH}" "dev-0" "${DATA_PATH}/glue-lm-gap"
    generate_lm_gap_data "${LM_GAP_CHALLENGE_PATH}" "test-A" "${DATA_PATH}/glue-lm-gap"
fi

# Fine-tuned models
DATA_SAVED_MODELS_PATH="out/glue/finetuned_model"
DATA_MODEL_PATH="${DATA_PATH}/finetuned_model"
generate_models_results "${DATA_SAVED_MODELS_PATH}" "${DATA_MODEL_PATH}" "validation"

DATA_LM_GAP_PATH="${DATA_PATH}/glue-lm-gap"
DATA_LOSS_PATH="${DATA_PATH}/loss"
# Draw one plot
log '=== Generating main plot ==='
TABLE_LABEL="table:glue_score_validation"
TABLE_CAPTION="Wyniki dostrojonych neuronowych modeli języka na zbiorze walidacyjnym na zbiorze zadań GLUE Benchmark."
generate_aggregated_results_with_lm_gap_and_loss "${DATA_MODEL_PATH}" "${DATA_LM_GAP_PATH}" "${DATA_LOSS_PATH}" "validation"
draw_plot "${DATA_MODEL_PATH}" "validation"
# Get all plots
GENERATE_ALL_PLOTS=0
# Skip generating if no needed
if [ "${GENERATE_ALL_PLOTS}" -gt 0 ]; then
    # Draw plot for each dataset separately
    for i_dataset_name in 'CoLA' 'MNLI-m' 'MNLI-mm' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST-2' 'STS-B'; do
        log "=== Generating plot for dataset: ${i_dataset_name} ==="
        DATASET_NAME="${i_dataset_name}" generate_aggregated_results_with_lm_gap_and_loss "${DATA_MODEL_PATH}" "${DATA_LM_GAP_PATH}" "${DATA_LOSS_PATH}" "validation"
        DATASET_NAME="${i_dataset_name}" draw_plot "${DATA_MODEL_PATH}" "validation"
    done
    # Draw one plot for each dataset LM-GAP score
    for i_dataset_name in 'CoLA' 'MNLI-m' 'MNLI-mm' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST-2' 'STS-B'; do
        log "=== Generating plot for LM-GAP score from: ${i_dataset_name} ==="
        LM_GAP_SCORE_KEY="${i_dataset_name}" generate_aggregated_results_with_lm_gap_and_loss "${DATA_MODEL_PATH}" "${DATA_LM_GAP_PATH}" "${DATA_LOSS_PATH}" "validation"
        LM_GAP_SCORE_KEY="${i_dataset_name}" draw_plot "${DATA_MODEL_PATH}" "validation"
    done
fi

# Generate correlation results for fine-tuned models
mkdir -p "${DATA_MODEL_PATH}/correlations"
# Loss correlations
TABLE_LABEL="table:glue_correlations_validation_loss"
TABLE_CAPTION="Korelacja wyników neuronowych modeli języka pomiędzy wynikiem funkcji kosztu oraz wynikami na zbiorze zadań GLUE Benchmark na zbiorze walidacyjnym."
compute_correlations_results "${DATA_MODEL_PATH}" "validation" "loss"
TABLE_LABEL="table:glue_correlations_validation_loss_encoder"
TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o koder architektury Transformer pomiędzy wynikiem funkcji kosztu oraz wynikami na zbiorze zadań GLUE Benchmark na zbiorze walidacyjnym."
compute_correlations_results "${DATA_MODEL_PATH}" "validation" "loss" "encoder"
TABLE_LABEL="table:glue_correlations_validation_loss_decoder"
TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o dekoder architektury Transformer pomiędzy wynikiem funkcji kosztu oraz wynikami na zbiorze zadań GLUE Benchmark na zbiorze walidacyjnym."
compute_correlations_results "${DATA_MODEL_PATH}" "validation" "loss" "decoder"
TABLE_LABEL="table:glue_correlations_validation_loss_encoder_decoder"
TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o koder-dekoder architektury Transformer pomiędzy wynikiem funkcji kosztu oraz wynikami na zbiorze zadań GLUE Benchmark na zbiorze walidacyjnym."
compute_correlations_results "${DATA_MODEL_PATH}" "validation" "loss" "encoder-decoder"
# LM-GAP correlations
TABLE_CAPTION="Korelacja wyników neuronowych modeli języka pomiędzy wynikiem zadania zgadywania zamaskowanego słowa oraz wynikami na zbiorze zadań GLUE Benchmark na zbiorze walidacyjnym."
TABLE_LABEL="table:glue_correlations_validation_lm_gap"
compute_correlations_results "${DATA_MODEL_PATH}" "validation" "LM-GAP"
TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o koder architektury Transformer pomiędzy wynikiem zadania zgadywania zamaskowanego słowa oraz wynikami na zbiorze zadań GLUE Benchmark na zbiorze walidacyjnym."
TABLE_LABEL="table:glue_correlations_validation_lm_gap_encoder"
compute_correlations_results "${DATA_MODEL_PATH}" "validation" "LM-GAP" "encoder"
TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o dekoder architektury Transformer pomiędzy wynikiem zadania zgadywania zamaskowanego słowa oraz wynikami na zbiorze zadań GLUE Benchmark na zbiorze walidacyjnym."
TABLE_LABEL="table:glue_correlations_validation_lm_gap_decoder"
compute_correlations_results "${DATA_MODEL_PATH}" "validation" "LM-GAP" "decoder"
TABLE_CAPTION="Korelacja wyników neuronowych modeli języka opartych o koder-dekoder architektury Transformer pomiędzy wynikiem zadania zgadywania zamaskowanego słowa oraz wynikami na zbiorze zadań GLUE Benchmark na zbiorze walidacyjnym."
TABLE_LABEL="table:glue_correlations_validation_lm_gap_encoder_decoder"
compute_correlations_results "${DATA_MODEL_PATH}" "validation" "LM-GAP" "encoder-decoder"

# Official models - data from publications or from GLUE Benchmark webpage
DATA_MODEL_PATH="${DATA_PATH}/official_model"

TABLE_LABEL="table:glue_score_official_validation"
TABLE_CAPTION="Wyniki neuronowych modeli języka na zbiorze walidacyjnym na zbiorze zadań GLUE Benchmark."
generate_aggregated_results "${DATA_MODEL_PATH}" "validation"
TABLE_LABEL="table:glue_score_official_test"
TABLE_CAPTION="Wyniki neuronowych modeli języka na zbiorze testowym na zbiorze zadań GLUE Benchmark."
generate_aggregated_results "${DATA_MODEL_PATH}" "test"

# Merged models - for easier comparison
DATA_MODEL_PATH="${DATA_PATH}/merged_model"
TABLE_LABEL="table:glue_merged_score_validation"
TABLE_CAPTION="Wyniki neuronowych modeli języka na zbiorze walidacyjnym na zbiorze zadań GLUE Benchmark."
generate_aggregated_results "${DATA_MODEL_PATH}" "validation"
