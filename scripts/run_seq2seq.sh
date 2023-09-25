#!/usr/bin/env bash

set -eu

function log() {
    date_now=$(date '+%FT%T')
    echo "[${date_now}] $1" | tee -a log-finetuning.log
}

if ! command -v jq &>/dev/null; then
    log '[ERROR] Require "jq" command line'
    exit 1
fi

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

# Get parameters file path
FILE_PARAMETERS_PATH="$(dirname "$(realpath "$0")")/parameters_fine_tuning.json"
FILE_SCORE_PATH="$(dirname "$(realpath "$0")")/expected_fine_tuning_score.json"

# Shared parameters
MAX_RETRIES="${MAX_RETRIES:=5}"
MODEL_SEED="${MODEL_SEED:=42}"
DATA_SEED="${DATA_SEED:=44}"

# Save directory
SAVE_PATH="${SAVE_PATH:="out/fine-tuning"}"
mkdir -p "${SAVE_PATH}"

function fine_tune_model {
    log '[INFO] === Start preparing fine-tuning ==='
    if [ "$#" -lt 4 ]; then
        log '[ERROR] Missing parameters in fine_tune_model function'
        exit 1
    fi
    challenge_path="$1"
    challenge_name=$(basename "$1")
    dataset_name="$2"
    model_name="$3"
    model_human_name="$4"
    model_run_id="${RUN_ID:=}"
    keep_model_checkpoints=${KEEP_MODEL_CHECKPOINTS:=1}

    # Set seeds and save path
    data_seed="${DATA_SEED}"
    if [ "${model_run_id}" == '' ]; then
        model_seed="${MODEL_SEED}"
        save_dir="${SAVE_PATH}/${model_human_name}-${dataset_name}"
    # If RUN_ID is exported it will be added to model seed and save path
    else
        model_seed=$((MODEL_SEED + model_run_id))
        save_dir="${SAVE_PATH}/${model_human_name}-${dataset_name}-${model_run_id}"
    fi

    log "[INFO] Preparing fine-tuning for challenge: \"${challenge_name}\", dataset: \"${dataset_name}\" and model: \"${model_name}\" (${model_human_name})"

    # Get parameters
    jq_query=".\"${challenge_name}\".\"${dataset_name}\""
    # Get epochs parameter
    parameter_epochs=${PARAMETER_EPOCHS:="null"}
    if [ "${parameter_epochs}" == 'null' ]; then
        parameter_epochs=$(jq -e -r "${jq_query}.epochs" "${FILE_PARAMETERS_PATH}" || true)
    fi
    # Get learning rate parameter
    parameter_lr=${PARAMETER_LR:="null"}
    if [ "${parameter_lr}" == 'null' ]; then
        parameter_lr=$(jq -e -r "${jq_query}.learning_rate" "${FILE_PARAMETERS_PATH}" || true)
    fi
    # Get evaluation steps
    evaluation_steps=$(jq -e -r "${jq_query}.evaluation_steps" "${FILE_PARAMETERS_PATH}" || true)
    if [ "${evaluation_steps}" == 'null' ]; then
        evaluation_steps='500'
    fi

    # Get other parameters
    parameter_sequence_length="${SEQUENCE_LENGTH:=256}"
    parameter_pad_to_max_length="${PAD_TO_MAX_LENGTH:="False"}"
    parameter_batch_size=${BATCH_SIZE:=8}

    # Log parameters
    log "[INFO] Using parameters:"
    log "       * batch_size       : ${parameter_batch_size}"
    log "       * epoch            : ${parameter_epochs}"
    log "       * learning_rate    : ${parameter_lr}"
    log "       * sequence_length  : ${parameter_sequence_length}"
    log "       * pad_to_max_length: ${parameter_pad_to_max_length}"
    log "       * evaluation_steps : ${evaluation_steps}"

    # Check parameters are valid
    if [ "${parameter_epochs}" == 'null' ] || [ "${parameter_lr}" == 'null' ]; then
        log '[ERROR] Invalid parameters, skipping fine-tuning'
        return
    fi

    # Get expected score
    expected_score=$(jq -e -r "${jq_query}.\"${model_name}\" | tonumber" "${FILE_SCORE_PATH}" || true)
    log "[INFO] Using expected score: ${expected_score}"
    if [ "${expected_score}" == 'null' ]; then
        log "[ERROR] Cannot found expected score, skipping fine-tuning"
        return
    fi

    counter_training=0
    while true; do
        counter_training=$((counter_training + 1))
        if [ "${counter_training}" -gt "${MAX_RETRIES}" ]; then
            log "[ERROR] Reach maximum number of training: ${MAX_RETRIES}, stop retrying!"
            break
        fi

        mkdir -p "${save_dir}"
        log "[INFO] Saving model path: ${save_dir}"
        log "[INFO] Using model seed: ${model_seed} and data seed: ${data_seed}"

        # Run fine-tuning
        log "[INFO] Running fine-tuning for challenge: ${challenge_path}, dataset: ${dataset_name} and model: ${model_name} (${model_human_name})"
        poetry run python -m phd_model_evaluations.cli.train.finetuning.run_finetuning_seq2seq \
            --train_file "${challenge_path}/train/dataset_data.json.xz" \
            --validation_file "${challenge_path}/test-A/dataset_data.json.xz" \
            --output_dir "${save_dir}" \
            --dataset_name "${dataset_name}" \
            --model_name "${model_name}" \
            --model_human_name "${model_human_name}" \
            --sequence_length "${parameter_sequence_length}" \
            --logging_strategy steps --logging_steps "${evaluation_steps}" \
            --save_strategy steps --save_steps "${evaluation_steps}" \
            --save_total_limit 2 \
            --evaluation_strategy steps --eval_steps "${evaluation_steps}" \
            --early_stopping_patience 8 \
            --early_stopping_threshold '0.00001' \
            --per_device_train_batch_size "${parameter_batch_size}" \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --max_batch_size_with_gradient_accumulation 32 \
            --seed "${model_seed}" \
            --data_seed "${data_seed}" \
            --num_train_epochs "${parameter_epochs}" \
            --max_validation_samples 2000 \
            --optim adamw_torch \
            --warmup_ratio '0.06' \
            --adam_epsilon '1e-6' \
            --adam_beta1 '0.9' \
            --adam_beta2 '0.98' \
            --learning_rate "${parameter_lr}" \
            --tf32 True \
            --log_file "${save_dir}/log-fine-tuning.log" \
            --pad_to_max_length "${parameter_pad_to_max_length}" \
            --verbose True

        # Get score from validation set
        score_file="${save_dir}/trainer_state.json"
        score_validation=$(jq -e -r '.best_metric' "${score_file}")
        log "[INFO] Achieve validation score: ${score_validation}"

        # Check if is empty score
        if (("$(echo "${score_validation} <= 0.0" | bc)")); then
            log "[ERROR] Detect empty validation metric! Retrying..."
            rm -rf "${save_dir}"
            data_seed=$((data_seed + 1))
            log "[INFO] Changed data seed: ${data_seed}"
            continue
        fi

        # Run evaluation
        log "[INFO] Running evaluation for challenge: ${challenge_path}, dataset: ${dataset_name} and model: ${model_name} (${model_human_name})"
        poetry run python -m phd_model_evaluations.cli.evaluation.model.evaluate_seq2seq_model \
            --test_file "${challenge_path}/test-A/dataset_data.json.xz" \
            --output_dir "${save_dir}" \
            --dataset_name "${dataset_name}" \
            --model_name "${save_dir}" \
            --model_human_name "${model_human_name}" \
            --find_best_model True \
            --sequence_length "${parameter_sequence_length}" \
            --seed "${model_seed}" \
            --data_seed "${data_seed}" \
            --tf32 True \
            --log_file "${save_dir}/log-fine-tuning.log" \
            --pad_to_max_length "${parameter_pad_to_max_length}" \
            --verbose True

        # Get score from final evaluation
        score_file="${save_dir}/evaluation_output.json"
        score=$(jq -e -r '.[0].best_metric' "${score_file}")
        log "[INFO] Achieve evaluation score: ${score}"

        log "[INFO] Validation score: ${score_validation} and evaluation score: ${score} and expected score: ${expected_score}"
        # Check if is empty score
        if (("$(echo "${score} <= 0.0" | bc)")); then
            log "[ERROR] Detect empty evaluation metric! Retrying..."
            rm -rf "${save_dir}"
            data_seed=$((data_seed + 1))
            log "[INFO] Changed data seed: ${data_seed}"
            continue

        # Check if score is in accepted range
        elif (("$(echo "${score} < ${expected_score}" | bc)")); then
            log "[ERROR] Expected score: ${expected_score} but got: ${score}"

        # Match expected score
        else
            log "[INFO] Reach expected score: ${expected_score}, achieved score: ${score}"
        fi

        break
    done

    if [ "${keep_model_checkpoints}" -eq 0 ]; then
        log "[INFO] Removing checkpoints ..."
        rm -rf "${save_dir}/checkpoint-"* || true
    fi
}

CHALLENGE_NAME='challenges/glue-lm-gap'
# | NAME  |  TRAIN   |  DEV-0  |  TEST-A |
# | ----- | -------- | ------- | ------- |
# | WNLI  |    1_104 |     122 |     140 |
# | RTE   |    4_449 |     496 |     553 |
# | MRPC  |    6_604 |     732 |     816 |
# | COLA  |    7_240 |     804 |     967 |
# | STS-B |   10_317 |   1_146 |   2_997 |
# | SST-2 |   43_928 |   4_879 |     869 |
# | QNLI  |  188_008 |  20_891 |  10_890 |
# | QQP   |  649_262 |  72_140 |  80_210 |
# | MNLI  |  686_797 |  76_311 |  19_250 |

# This is a function to fine-tuning many models
function fine_tune_models() {
    for dataset_name in 'rte' 'mrpc' 'cola' 'stsb' 'sst2' 'qnli' 'qqp' 'mnli'; do
        # Encoder-Decoder models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 't5-small' 'T5-small'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 't5-base' 'T5-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/t5-v1_1-small' 'T5-small-v1.1'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/t5-v1_1-base' 'T5-base-v1.1'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/t5-small-lm-adapt' 'T5-small-v1.1-lm-adapt'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/t5-base-lm-adapt' 'T5-base-v1.1-lm-adapt'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/t5-efficient-tiny' 'T5-efficient-tiny'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/t5-efficient-mini' 'T5-efficient-mini'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/t5-efficient-small' 'T5-efficient-small'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/t5-efficient-base' 'T5-efficient-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/switch-base-8' 'Switch-base-8'
        # Encoder-Decoder multi language models #
        if [ "${dataset_name}" == 'stsb' ] || [ "${dataset_name}" == 'cola' ]; then
            PARAMETER_LR='3e-4' fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/mt5-small' 'mT5-small'
            PARAMETER_LR='3e-4' fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/mt5-base' 'mT5-base'
        else
            fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/mt5-small' 'mT5-small'
            fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/mt5-base' 'mT5-base'
        fi
        SEQUENCE_LENGTH='512' BATCH_SIZE=4 fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/byt5-small' 'ByT5-small'
        SEQUENCE_LENGTH='512' BATCH_SIZE=2 fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/byt5-base' 'ByT5-base'
        # Encoder-Decoder long sequence models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/long-t5-tglobal-base' 'LongT5-TGlobal-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/long-t5-local-base' 'LongT5-Local-base'
        # Encoder-Decoder few-shot models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/flan-t5-small' 'FLAN-T5-small'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/flan-t5-base' 'FLAN-T5-base'
    done
}
# Uncomment to run fine-tuning models
export KEEP_MODEL_CHECKPOINTS="0"
fine_tune_models
