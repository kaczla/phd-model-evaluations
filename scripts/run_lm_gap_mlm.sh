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
    log '[INFO] === Start preparing MLM LM-GAP ==='
    if [ "$#" -lt 4 ]; then
        log '[ERROR] Missing parameters in evaluate_model function'
        exit 1
    fi
    challenge_name="$1"
    set_name="$2"
    model_name="$3"
    model_human_name="$4"

    # Get other parameters
    parameter_sequence_length="${SEQUENCE_LENGTH:="None"}"
    parameter_add_prefix_space="${ADD_PREFIX_SPACE:="False"}"

    # Move optional arguments to special variable
    extra_parameters=()
    if [ "${parameter_sequence_length}" != 'None' ]; then
        extra_parameters=("${extra_parameters[@]}" "--sequence_length" "${parameter_sequence_length}")
    fi

    # Log parameters
    log "[INFO] Using parameters:"
    log "       * sequence_length  : ${parameter_sequence_length}"
    log "       * add_prefix_space : ${parameter_add_prefix_space}"
    log "       = extra_parameters : " "${extra_parameters[@]}"

    log "[INFO] Running LM-GAP for challenge: ${challenge_name}/${set_name} and model: ${model_name} (${model_human_name})"
    poetry run python -m phd_model_evaluations.cli.evaluation.lm_gap.evaluate_mlm_model \
        --file_in "${challenge_name}/${set_name}/in.tsv.xz" \
        --file_out "${challenge_name}/${set_name}" \
        --model_name "${model_name}" \
        --model_human_name "${model_human_name}" \
        --generate_out_file_name True \
        --method 'simple' \
        --add_prefix_space "${parameter_add_prefix_space}" \
        "${extra_parameters[@]}" \
        --verbose 'True'
}

CHALLENGE_NAME='challenges/glue-lm-gap'

for set_name in 'dev-0' 'test-A'; do
    # Encoder models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'roberta-base' 'RoBERTa-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'roberta-large' 'RoBERTa-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'bert-base-uncased' 'BERT-base-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'bert-base-cased' 'BERT-base-cased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'bert-large-uncased' 'BERT-large-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'bert-large-cased' 'BERT-large-cased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'prajjwal1/bert-tiny' 'BERT-tiny-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'prajjwal1/bert-mini' 'BERT-mini-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'prajjwal1/bert-small' 'BERT-small-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'prajjwal1/bert-medium' 'BERT-medium-uncased'
    # Encoder distilled models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'distilroberta-base' 'DistilRoBERTa-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'distilbert-base-uncased' 'DistilBERT-base-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'distilbert-base-cased' 'DistilBERT-base-cased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'albert-base-v2' 'ALBERT-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'albert-large-v2' 'ALBERT-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'albert-xlarge-v2' 'ALBERT-xlarge'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'albert-xxlarge-v2' 'ALBERT-xxlarge'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'google/mobilebert-uncased' 'MobileBERT-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L12-H384-distilled-from-RoBERTa-Large' 'MiniLM-L12-H384-RoBERTa-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large' 'MiniLM-L6-H768-RoBERTa-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large' 'MiniLM-L6-H384-RoBERTa-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base' 'MiniLM-L6-H768-BERT-base-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base' 'MiniLM-L6-H384-BERT-base-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large' 'MiniLM-L6-H768-BERT-large-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large' 'MiniLM-L6-H384-BERT-large-uncased'
    # Encoder multi language models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'xlm-roberta-base' 'XLM-RoBERTa-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'xlm-roberta-large' 'XLM-RoBERTa-large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'xlm-mlm-100-1280' 'XLM-100-lang'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'xlm-mlm-17-1280' 'XLM-17-lang'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'xlm-mlm-en-2048' 'XLM-en'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'bert-base-multilingual-uncased' 'BERT-base-multilingual-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'bert-base-multilingual-cased' 'BERT-base-multilingual-cased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large' 'MiniLM-L12-H384-XLMR-Large'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large' 'MiniLM-L6-H384-XLMR-Large'
    # Encoder non english models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'camembert-base' 'CamemBERT-base'
    SEQUENCE_LENGTH=512 ADD_PREFIX_SPACE="True" evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'sdadas/polish-roberta-base-v1' 'PolishRoBERT-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'bert-base-german-cased' 'German-BERT-base-cased'
    # Encoder domain models #
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'yiyanghkust/finbert-pretrain' 'FinBERT'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'allenai/scibert_scivocab_uncased' 'SciBERT-uncased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'allenai/scibert_scivocab_cased' 'SciBERT-cased'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'allenai/biomed_roberta_base' 'BioMed-RoBERTa-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'microsoft/codebert-base-mlm' 'CodeBERT-base'
    evaluate_model "${CHALLENGE_NAME}" "${set_name}" 'emilyalsentzer/Bio_ClinicalBERT' 'ClinicalBERT'
done
