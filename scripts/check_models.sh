#!/usr/bin/env bash

set -eu

function log() {
    date_now=$(date '+%FT%T')
    echo "[${date_now}] $1" | tee -a log-check-models.log
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

function check_model {
    log '[INFO] === Start checking model ==='
    if [ "$#" -lt 1 ]; then
        log '[ERROR] Missing parameters in check_model function'
        exit 1
    fi
    model_name="$1"

    log "[INFO] Checking model: ${model_name}"
    poetry run python -m phd_model_evaluations.cli.evaluation.check_model \
        --model_name "${model_name}" \
        --verbose True
    log "[INFO] Model: ${model_name} valid"
}

check_model 'prajjwal1/bert-tiny'
