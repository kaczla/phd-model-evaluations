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
    log '[INFO] === Start preparing CLM LM-GAP ==='
    if [ "$#" -lt 4 ]; then
        log '[ERROR] Missing parameters in evaluate_model function'
        exit 1
    fi
    challenge_name="$1"
    set_name="$2"
    model_name="$3"
    model_human_name="$4"

    log "[INFO] Running LM-GAP for challenge: ${challenge_name}/${set_name} and model: ${model_name} (${model_human_name})"
    poetry run python -m phd_model_evaluations.cli.evaluation.lm_gap.evaluate_clm_model \
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
    # Decoder models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'gpt2' 'GPT-2-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'gpt2-medium' 'GPT-2-medium'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'gpt2-large' 'GPT-2-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/gpt-neo-125M' 'GPT-Neo-125M'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-70m' 'Pythia-70M'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-70m-deduped' 'Pythia-70M-deduped'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-160m' 'Pythia-160M'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-160m-deduped' 'Pythia-160M-deduped'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-410m' 'Pythia-410M'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-410m-deduped' "Pythia-410M-deduped"
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'facebook/opt-125m' 'OPT-125M'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'facebook/opt-350m' 'OPT-350M'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'cerebras/Cerebras-GPT-111M' 'Cerebras-GPT-111M'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'cerebras/Cerebras-GPT-256M' 'Cerebras-GPT-256M'
    # Encoder distilled models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'distilgpt2' 'DistilGPT-2'
    # Decoder domain models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'microsoft/biogpt' 'BioGPT'
    # Decoder non english models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'asi/gpt-fr-cased-small' 'GPT-fr-small'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'asi/gpt-fr-cased-base' 'GPT-fr-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'sdadas/polish-gpt2-small' 'PolishGPT-2-small'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'sdadas/polish-gpt2-medium' 'PolishGPT-2-medium'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'sdadas/polish-gpt2-large' 'PolishGPT-2-large'
done
