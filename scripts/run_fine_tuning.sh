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

    # Get other parameters
    parameter_sequence_length="${SEQUENCE_LENGTH:=256}"
    parameter_pad_to_max_length="${PAD_TO_MAX_LENGTH:="False"}"
    parameter_batch_size=${BATCH_SIZE:=8}
    parameter_use_custom_model="${CUSTOM_MODEL:="False"}"

    # Log parameters
    log "[INFO] Using parameters:"
    log "       * batch_size       : ${parameter_batch_size}"
    log "       * epoch            : ${parameter_epochs}"
    log "       * learning_rate    : ${parameter_lr}"
    log "       * sequence_length  : ${parameter_sequence_length}"
    log "       * pad_to_max_length: ${parameter_pad_to_max_length}"
    log "       * use_custom_model : ${parameter_use_custom_model}"

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
        poetry run python -m phd_model_evaluations.cli.train.finetuning.run_finetuning \
            --train_file "${challenge_path}/train/dataset_data.json.xz" \
            --validation_file "${challenge_path}/test-A/dataset_data.json.xz" \
            --output_dir "${save_dir}" \
            --dataset_name "${dataset_name}" \
            --model_name "${model_name}" \
            --model_human_name "${model_human_name}" \
            --sequence_length "${parameter_sequence_length}" \
            --logging_strategy steps --logging_steps 500 \
            --save_strategy steps --save_steps 500 \
            --save_total_limit 2 \
            --evaluation_strategy steps --eval_steps 500 \
            --early_stopping_patience 8 \
            --early_stopping_threshold '0.00001' \
            --per_device_train_batch_size "${parameter_batch_size}" \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --max_batch_size_with_gradient_accumulation 32 \
            --seed "${model_seed}" \
            --data_seed "${data_seed}" \
            --num_train_epochs "${parameter_epochs}" \
            --max_validation_samples 5000 \
            --optim adamw_torch \
            --warmup_ratio '0.06' \
            --adam_epsilon '1e-6' \
            --adam_beta1 '0.9' \
            --adam_beta2 '0.98' \
            --learning_rate "${parameter_lr}" \
            --tf32 True \
            --log_file "${save_dir}/log-fine-tuning.log" \
            --pad_to_max_length "${parameter_pad_to_max_length}" \
            --use_custom_model "${parameter_use_custom_model}" \
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
        poetry run python -m phd_model_evaluations.cli.evaluation.model.evaluate_model \
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
        # Encoder models #
        export KEEP_MODEL_CHECKPOINTS="1"
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'roberta-base' 'RoBERTa-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'roberta-large' 'RoBERTa-large'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'bert-base-cased' 'BERT-base-cased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'bert-base-uncased' 'BERT-base-uncased'
        export KEEP_MODEL_CHECKPOINTS="0"
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'prajjwal1/bert-tiny' 'BERT-tiny-uncased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'prajjwal1/bert-mini' 'BERT-mini-uncased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'prajjwal1/bert-small' 'BERT-small-uncased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'prajjwal1/bert-medium' 'BERT-medium-uncased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'bert-large-cased' 'BERT-large-cased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'bert-large-uncased' 'BERT-large-uncased'
        # Encoder distilled models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'distilroberta-base' 'DistilRoBERTa-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'distilbert-base-cased' 'DistilBERT-base-cased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'distilbert-base-uncased' 'DistilBERT-base-uncased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'albert-base-v2' 'ALBERT-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'albert-large-v2' 'ALBERT-large'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'google/mobilebert-uncased' 'MobileBERT-uncased'
        # Encoder domain models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'yiyanghkust/finbert-pretrain' 'FinBERT'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'allenai/scibert_scivocab_uncased' 'SciBERT-uncased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'allenai/scibert_scivocab_cased' 'SciBERT-cased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'allenai/biomed_roberta_base' 'BioMed-RoBERTa-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'emilyalsentzer/Bio_ClinicalBERT' 'ClinicalBERT'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'microsoft/codebert-base-mlm' 'CodeBERT-base'
        # Encoder long sequence models #
        PAD_TO_MAX_LENGTH="True" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'allenai/longformer-base-4096' 'Longformer-base'
        PAD_TO_MAX_LENGTH="True" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'allenai/longformer-large-4096' 'Longformer-large'
        # Encoder multi language models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'xlm-roberta-base' 'XLM-RoBERTa-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'xlm-roberta-large' 'XLM-RoBERTa-large'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'bert-base-multilingual-uncased' 'BERT-base-multilingual-uncased'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'bert-base-multilingual-cased' 'BERT-base-multilingual-cased'
        # Encoder non english models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'camembert-base' 'CamemBERT-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'sdadas/polish-roberta-base-v1' 'PolishRoBERT-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'bert-base-german-cased' 'German-BERT-base-cased'
        # Decoder models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'gpt2' 'GPT-2-base'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'gpt2-medium' 'GPT-2-medium'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'EleutherAI/gpt-neo-125M' 'GPT-Neo-125M'
        CUSTOM_MODEL="True" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'EleutherAI/pythia-70m' 'Pythia-70M'
        CUSTOM_MODEL="True" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'EleutherAI/pythia-70m-deduped' 'Pythia-70M-deduped'
        CUSTOM_MODEL="True" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'EleutherAI/pythia-160m' 'Pythia-160M'
        CUSTOM_MODEL="True" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'EleutherAI/pythia-160m-deduped' 'Pythia-160M-deduped'
        CUSTOM_MODEL="True" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'EleutherAI/pythia-410m' 'Pythia-410M'
        CUSTOM_MODEL="True" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'EleutherAI/pythia-410m-deduped' 'Pythia-410M-deduped'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'facebook/opt-125m' 'OPT-125M'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'facebook/opt-350m' 'OPT-350M'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'cerebras/Cerebras-GPT-111M' 'Cerebras-GPT-111M'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'cerebras/Cerebras-GPT-256M' 'Cerebras-GPT-256M'
        # Encoder distilled models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'distilgpt2' 'DistilGPT-2'
        # Decoder domain models #
        CUSTOM_MODEL="True" SEQUENCE_LENGTH=384 fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'microsoft/biogpt' 'BioGPT'
        # Decoder non english models #
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'asi/gpt-fr-cased-small' 'GPT-fr-small'
        fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" 'sdadas/polish-gpt2-small' 'PolishGPT-2-small'
    done
}
# Uncomment to run fine-tuning models
export KEEP_MODEL_CHECKPOINTS="0"
fine_tune_models

# This is a function to fine-tuning one model with different model seeds
function fine_tune_model_with_different_seeds() {
    model_name="roberta-base"
    model_human_name="RoBERTa-base"
    for dataset_name in 'rte' 'mrpc' 'cola' 'stsb' 'sst2' 'qnli' 'qqp' 'mnli'; do
        RUN_ID='' fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" "${model_name}" "${model_human_name}"
        RUN_ID='100' fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" "${model_name}" "${model_human_name}"
        RUN_ID='200' fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" "${model_name}" "${model_human_name}"
        RUN_ID='300' fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" "${model_name}" "${model_human_name}"
        RUN_ID='400' fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" "${model_name}" "${model_human_name}"
    done
}
# Uncomment to run fine-tuning one model with different model seeds
#fine_tune_model_with_different_seeds

# This is a function to fine-tuning one model with different parameters
function fine_tune_model_with_parameters_search() {
    model_name="roberta-base"
    model_human_name="RoBERTa-base"
    export KEEP_MODEL_CHECKPOINTS="0"

    for dataset_name in 'rte' 'mrpc' 'cola' 'stsb' 'sst2'; do
        counter=0
        for epochs in '3' '5' '10'; do
            for lr in '1e-5' '2e-5' '3e-5' '5e-5' '8e-5'; do
                counter=$((counter + 1))
                source_save_path="${SAVE_PATH}/${model_human_name}-${dataset_name}"
                target_save_path="${SAVE_PATH}/${model_human_name}-${dataset_name}-${counter}"
                if [ -d "${target_save_path}" ]; then
                    log "[INFO] Skip existing: ${target_save_path}"
                    continue
                fi
                log "[INFO] Running ${counter} fine-tuning with: epochs=${epochs} | learning rate=${lr}"
                # Run fine-tuning
                PARAMETER_EPOCHS="${epochs}" PARAMETER_LR="${lr}" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" "${model_name}" "${model_human_name}"
                if [ ! -d "${source_save_path}" ]; then
                    log "[ERROR] Cannot found model for: epochs=${epochs} | learning rate=${lr} | directory=${target_save_path}"
                    continue
                fi
                mv -v "${source_save_path}" "${target_save_path}"
                # Save statistics to file
                score=$(jq -r ".[0].best_metric" "${target_save_path}/evaluation_output.json")
                checkpoint_name=$(basename "$(jq -r ".best_model_checkpoint" "${target_save_path}/trainer_state.json")")
                printf '%s | %-8s' "${model_human_name}" "${dataset_name}-${counter}" | tee -a log-parameters-search.log
                printf ' | epochs=%-2s | learning_rate=%4s' "${epochs}" "${lr}" | tee -a log-parameters-search.log
                printf ' | score=%-18s | best_checkpoint=%s\n' "${score}" "${checkpoint_name}" | tee -a log-parameters-search.log
                printf '{"run_id": %d, "model_name": "%s", "model_human_name": "%s", "epochs": "%s", "learning_rate": "%s", "score": "%s", "best_checkpoint": "%s"}' "${counter}" "${model_name}" "${model_human_name}" "${epochs}" "${lr}" "${score}" "${checkpoint_name}" | jq > "${target_save_path}/search_parameters.json"
            done
        done
    done

    for dataset_name in 'qnli' 'qqp' 'mnli'; do
        counter=0
        for epochs in '3' '5'; do
            for lr in '2e-5' '5e-5' '8e-5'; do
                counter=$((counter + 1))
                source_save_path="${SAVE_PATH}/${model_human_name}-${dataset_name}"
                target_save_path="${SAVE_PATH}/${model_human_name}-${dataset_name}-${counter}"
                if [ -d "${target_save_path}" ]; then
                    log "[INFO] Skip existing: ${target_save_path}"
                    continue
                fi
                log "[INFO] Running ${counter} fine-tuning with: epochs=${epochs} | learning rate=${lr}"
                # Run fine-tuning
                PARAMETER_EPOCHS="${epochs}" PARAMETER_LR="${lr}" fine_tune_model "${CHALLENGE_NAME}" "${dataset_name}" "${model_name}" "${model_human_name}"
                if [ ! -d "${source_save_path}" ]; then
                    log "[ERROR] Cannot found model for: epochs=${epochs} | learning rate=${lr} | directory=${target_save_path}"
                    continue
                fi
                mv -v "${source_save_path}" "${target_save_path}"
                # Save statistics to file
                score=$(jq -r ".[0].best_metric" "${target_save_path}/evaluation_output.json")
                checkpoint_name=$(basename "$(jq -r ".best_model_checkpoint" "${target_save_path}/trainer_state.json")")
                printf '%s | %-8s' "${model_human_name}" "${dataset_name}-${counter}" | tee -a log-parameters-search.log
                printf ' | epochs=%-2s | learning_rate=%4s' "${epochs}" "${lr}" | tee -a log-parameters-search.log
                printf ' | score=%-18s | best_checkpoint=%s\n' "${score}" "${checkpoint_name}" | tee -a log-parameters-search.log
                printf '{"run_id": %d, "model_name": "%s", "model_human_name": "%s", "epochs": "%s", "learning_rate": "%s", "score": "%s", "best_checkpoint": "%s"}' "${counter}" "${model_name}" "${model_human_name}" "${epochs}" "${lr}" "${score}" "${checkpoint_name}" | jq > "${target_save_path}/search_parameters.json"
            done
        done
    done
}
# Uncomment to run fine-tuning one model with different parameters
#fine_tune_model_with_parameters_search
