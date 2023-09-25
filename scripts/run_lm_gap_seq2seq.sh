#!/usr/bin/env bash

set -eu

function log() {
    date_now=$(date '+%FT%T')
    echo "[${date_now}] $1" | tee -a log-lm-gap.log
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

function evaluate_model {
    log '[INFO] === Start preparing seq2seq LM-GAP ==='
    if [ "$#" -lt 4 ]; then
        echo >&2 '[ERROR] Missing parameters in evaluate_model function'
        exit 1
    fi
    challenge_name="$1"
    set_name="$2"
    model_name="$3"
    model_human_name="$4"

    log "[LOG] Running LM-GAP for challenge: ${challenge_name}/${set_name} and model: ${model_name} (${model_human_name})"
    poetry run python -m phd_model_evaluations.cli.evaluation.lm_gap.evaluate_seq2seq_model \
        --file_in "${challenge_name}/${set_name}/in.tsv.xz" \
        --file_out "${challenge_name}/${set_name}" \
        --model_name "${model_name}" \
        --model_human_name "${model_human_name}" \
        --generate_out_file_name True \
        --depth 1 \
        --verbose 'True'
}

CHALLENGE_NAME='challenges/glue-lm-gap'

for set_name in 'dev-0' 'test-A'; do
    # Encoder-Decoder models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 't5-small' 'T5-small'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 't5-base' 'T5-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 't5-large' 'T5-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-v1_1-small' 'T5-small-v1.1'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-v1_1-base' 'T5-base-v1.1'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-v1_1-large' 'T5-large-v1.1'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-small-lm-adapt' 'T5-small-v1.1-lm-adapt'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-base-lm-adapt' 'T5-base-v1.1-lm-adapt'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-large-lm-adapt' 'T5-large-v1.1-lm-adapt'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-tiny' 'T5-efficient-tiny'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-mini' 'T5-efficient-mini'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-small' 'T5-efficient-small'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-base' 'T5-efficient-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-large' 'T5-efficient-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/switch-base-8' 'Switch-base-8'
    # Encoder-Decoder distilled models #
    # None
    # Encoder-Decoder multi language models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/mt5-small' 'mT5-small'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/mt5-base' 'mT5-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/mt5-large' 'mT5-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/byt5-small' 'ByT5-small'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/byt5-base' 'ByT5-base'
    # Encoder-Decoder long sequence models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/long-t5-tglobal-base' 'LongT5-TGlobal-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/long-t5-local-base' 'LongT5-Local-base'
    # Encoder-Decoder few-shot models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/flan-t5-small' 'FLAN-T5-small'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/flan-t5-base' 'FLAN-T5-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/flan-t5-large' 'FLAN-T5-large'
done
