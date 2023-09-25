#!/usr/bin/env bash

set -eu

function log() {
    date_now=$(date '+%FT%T')
    echo "[${date_now}] $1" | tee -a log-loss.log
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

function compute_loss {
    log '[INFO] === Start preparing computing loss ==='
    if [ "$#" -lt 4 ]; then
        log '[ERROR] Missing parameters in compute_loss function'
        exit 1
    fi
    challenge_name="$1"
    set_name="$2"
    model_name="$3"
    model_human_name="$4"

    # Get save file name
    if [ "${set_name}" == 'dev-0' ]; then
        save_file_name='part-train.json'
    elif [ "${set_name}" == 'test-A' ]; then
        save_file_name='validation.json'
    else
        log "[ERROR] Cannot detect save file name for set name: ${set_name}"
        exit 2
    fi

    BATCH_SIZE="${BATCH_SIZE:=5}"

    log "[INFO] Running LM-GAP for challenge: ${challenge_name}/${set_name} and model: ${model_name} (${model_human_name}) with batch size: ${BATCH_SIZE}"
    poetry run python -m phd_model_evaluations.cli.evaluation.model.compute_loss \
        --file_path "${challenge_name}/${set_name}/raw_data.txt.xz" \
        --save_path "${SAVE_PATH}/${save_file_name}" \
        --model_name "${model_name}" \
        --model_human_name "${model_human_name}" \
        --join_examples True \
        --sequence_length 512 \
        --batch_size "${BATCH_SIZE}" \
        --device 0
}

CHALLENGE_NAME='challenges/glue-lm-gap'
SAVE_PATH="scripts/results/glue/loss"

for set_name in 'dev-0' 'test-A'; do
    # Encoder models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'roberta-base' 'RoBERTa-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'roberta-large' 'RoBERTa-large'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'bert-base-uncased' 'BERT-base-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'bert-base-cased' 'BERT-base-cased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'bert-large-uncased' 'BERT-large-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'bert-large-cased' 'BERT-large-cased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'prajjwal1/bert-tiny' 'BERT-tiny-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'prajjwal1/bert-mini' 'BERT-mini-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'prajjwal1/bert-small' 'BERT-small-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'prajjwal1/bert-medium' 'BERT-medium-uncased'
    # Encoder distilled models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'distilroberta-base' 'DistilRoBERTa-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'distilbert-base-uncased' 'DistilBERT-base-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'distilbert-base-cased' 'DistilBERT-base-cased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'albert-base-v2' 'ALBERT-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'albert-large-v2' 'ALBERT-large'
    BATCH_SIZE='2' compute_loss "${CHALLENGE_NAME}" "${set_name}" 'albert-xlarge-v2' 'ALBERT-xlarge'
    BATCH_SIZE='2' compute_loss "${CHALLENGE_NAME}" "${set_name}" 'albert-xxlarge-v2' 'ALBERT-xxlarge'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/mobilebert-uncased' 'MobileBERT-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L12-H384-distilled-from-RoBERTa-Large' 'MiniLM-L12-H384-RoBERTa-large'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large' 'MiniLM-L6-H768-RoBERTa-large'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large' 'MiniLM-L6-H384-RoBERTa-large'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base' 'MiniLM-L6-H768-BERT-base-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base' 'MiniLM-L6-H384-BERT-base-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large' 'MiniLM-L6-H768-BERT-large-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large' 'MiniLM-L6-H384-BERT-large-uncased'
    # Encoder multi language models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'xlm-roberta-base' 'XLM-RoBERTa-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'xlm-roberta-large' 'XLM-RoBERTa-large'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'bert-base-multilingual-uncased' 'BERT-base-multilingual-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'bert-base-multilingual-cased' 'BERT-base-multilingual-cased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large' 'MiniLM-L12-H384-XLMR-Large'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large' 'MiniLM-L6-H384-XLMR-Large'
    # Encoder domain models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'yiyanghkust/finbert-pretrain' 'FinBERT'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'allenai/scibert_scivocab_uncased' 'SciBERT-uncased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'allenai/scibert_scivocab_cased' 'SciBERT-cased'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'allenai/biomed_roberta_base' 'BioMed-RoBERTa-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'emilyalsentzer/Bio_ClinicalBERT' 'ClinicalBERT'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'microsoft/codebert-base-mlm' 'CodeBERT-base'
    # Encoder long sequence models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'allenai/longformer-base-4096' 'Longformer-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'allenai/longformer-large-4096' 'Longformer-large'
    # Encoder non english models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'camembert-base' 'CamemBERT-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'sdadas/polish-roberta-base-v1' 'PolishRoBERT-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'bert-base-german-cased' 'German-BERT-base-cased'
    # Decoder models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'gpt2' 'GPT-2-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'gpt2-medium' 'GPT-2-medium'
    BATCH_SIZE='2' compute_loss "${CHALLENGE_NAME}" "${set_name}" 'gpt2-large' 'GPT-2-large'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/gpt-neo-125M' 'GPT-Neo-125M'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-70m' 'Pythia-70M'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-70m-deduped' 'Pythia-70M-deduped'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-160m' 'Pythia-160M'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-160m-deduped' 'Pythia-160M-deduped'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-410m' 'Pythia-410M'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'EleutherAI/pythia-410m-deduped' "Pythia-410M-deduped"
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'facebook/opt-125m' 'OPT-125M'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'facebook/opt-350m' 'OPT-350M'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'cerebras/Cerebras-GPT-111M' 'Cerebras-GPT-111M'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'cerebras/Cerebras-GPT-256M' 'Cerebras-GPT-256M'
    # Encoder distilled models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'distilgpt2' 'DistilGPT-2'
    # Decoder domain models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'microsoft/biogpt' 'BioGPT'
    # Decoder non english models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'dbmdz/german-gpt2' 'German-GPT-2'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'stefan-it/german-gpt2-larger' 'German-GPT-2-larger'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'asi/gpt-fr-cased-small' 'GPT-fr-small'
    BATCH_SIZE='2' compute_loss "${CHALLENGE_NAME}" "${set_name}" 'asi/gpt-fr-cased-base' 'GPT-fr-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'sdadas/polish-gpt2-small' 'PolishGPT-2-small'
    BATCH_SIZE='2' compute_loss "${CHALLENGE_NAME}" "${set_name}" 'sdadas/polish-gpt2-medium' 'PolishGPT-2-medium'
    BATCH_SIZE='2' compute_loss "${CHALLENGE_NAME}" "${set_name}" 'sdadas/polish-gpt2-large' 'PolishGPT-2-large'
    # Encoder-Decoder models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 't5-small' 'T5-small'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 't5-base' 'T5-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/t5-v1_1-small' 'T5-small-v1.1'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/t5-v1_1-base' 'T5-base-v1.1'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/t5-small-lm-adapt' 'T5-small-v1.1-lm-adapt'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/t5-base-lm-adapt' 'T5-base-v1.1-lm-adapt'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-tiny' 'T5-efficient-tiny'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-mini' 'T5-efficient-mini'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-small' 'T5-efficient-small'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/t5-efficient-base' 'T5-efficient-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/switch-base-8' 'Switch-base-8'
    # Encoder-Decoder multi language models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/mt5-small' 'mT5-small'
    BATCH_SIZE='2' compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/mt5-base' 'mT5-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/byt5-small' 'ByT5-small'
    BATCH_SIZE='2' compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/byt5-base' 'ByT5-base'
    # Encoder-Decoder long sequence models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/long-t5-tglobal-base' 'LongT5-TGlobal-base'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/long-t5-local-base' 'LongT5-Local-base'
    # Encoder-Decoder few-shot models #
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/flan-t5-small' 'FLAN-T5-small'
    compute_loss "${CHALLENGE_NAME}" "${set_name}" 'google/flan-t5-base' 'FLAN-T5-base'
done
