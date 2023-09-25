from typing import List, Tuple

DEFAULT_ENCODER_MODEL_NAMES_AND_HUMAN_NAMES: List[Tuple[str, str]] = [
    # English encoder
    ("roberta-base", "RoBERTa-base"),
    ("roberta-large", "RoBERTa-large"),
    ("bert-base-cased", "BERT-base-cased"),
    ("bert-base-uncased", "BERT-base-uncased"),
    ("bert-large-cased", "BERT-large-cased"),
    ("bert-large-uncased", "BERT-large-uncased"),
    ("prajjwal1/bert-tiny", "BERT-tiny-uncased"),
    ("prajjwal1/bert-mini", "BERT-mini-uncased"),
    ("prajjwal1/bert-small", "BERT-small-uncased"),
    ("prajjwal1/bert-medium", "BERT-medium-uncased"),
    # English distilled encoder
    ("distilroberta-base", "DistilRoBERTa-base"),
    ("distilbert-base-cased", "DistilBERT-base-cased"),
    ("distilbert-base-uncased", "DistilBERT-base-uncased"),
    ("albert-base-v2", "ALBERT-base"),
    ("albert-large-v2", "ALBERT-large"),
    ("google/mobilebert-uncased", "MobileBERT-uncased"),
    # ("nreimers/MiniLMv2-L12-H384-distilled-from-RoBERTa-Large", "MiniLM-L12-H384-RoBERTa-large"),
    # ("nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large", "MiniLM-L6-H768-RoBERTa-large"),
    # ("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large", "MiniLM-L6-H384-RoBERTa-large"),
    # ("nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base", "MiniLM-L6-H768-BERT-base-uncased"),
    # ("nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base", "MiniLM-L6-H384-BERT-base-uncased"),
    # ("nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large", "MiniLM-L6-H768-BERT-large-uncased"),
    # ("nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large", "MiniLM-L6-H384-BERT-large-uncased"),
    # Domain models
    ("yiyanghkust/finbert-pretrain", "FinBERT"),
    ("allenai/scibert_scivocab_uncased", "SciBERT-uncased"),
    ("allenai/scibert_scivocab_cased", "SciBERT-cased"),
    ("allenai/biomed_roberta_base", "BioMed-RoBERTa-base"),
    ("emilyalsentzer/Bio_ClinicalBERT", "ClinicalBERT"),
    ("microsoft/codebert-base-mlm", "CodeBERT-base"),
    # Multilingual encoder
    ("xlm-roberta-base", "XLM-RoBERTa-base"),
    ("xlm-roberta-large", "XLM-RoBERTa-large"),
    ("bert-base-multilingual-uncased", "BERT-base-multilingual-uncased"),
    ("bert-base-multilingual-cased", "BERT-base-multilingual-cased"),
    # ("nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large", "MiniLM-L12-H384-XLMR-Large"),
    # ("nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large", "MiniLM-L6-H384-XLMR-Large"),
    # Long sequence encoder
    ("allenai/longformer-base-4096", "Longformer-base"),
    ("allenai/longformer-large-4096", "Longformer-large"),
    # Non english encoder
    ("camembert-base", "CamemBERT-base"),
    ("sdadas/polish-roberta-base-v1", "PolishRoBERT-base"),
    ("bert-base-german-cased", "German-BERT-base-cased"),
]

DEFAULT_ENCODER_HUMAN_MODEL_NAME_SET = {
    human_model_name for _, human_model_name in DEFAULT_ENCODER_MODEL_NAMES_AND_HUMAN_NAMES
}

DEFAULT_DECODER_MODEL_NAMES_AND_HUMAN_NAMES: List[Tuple[str, str]] = [
    # English decoder
    ("gpt2", "GPT-2-base"),
    ("gpt2-medium", "GPT-2-medium"),
    ("EleutherAI/gpt-neo-125M", "GPT-Neo-125M"),
    ("EleutherAI/pythia-70m", "Pythia-70M"),
    ("EleutherAI/pythia-70m-deduped", "Pythia-70M-deduped"),
    ("EleutherAI/pythia-160m", "Pythia-160M"),
    ("EleutherAI/pythia-160m-deduped", "Pythia-160M-deduped"),
    ("EleutherAI/pythia-410m", "Pythia-410M"),
    ("EleutherAI/pythia-410m-deduped", "Pythia-410M-deduped"),
    ("facebook/opt-125m", "OPT-125M"),
    ("facebook/opt-350m", "OPT-350M"),
    ("cerebras/Cerebras-GPT-111M", "Cerebras-GPT-111M"),
    ("cerebras/Cerebras-GPT-256M", "Cerebras-GPT-256M"),
    # English distilled decoder
    ("distilgpt2", "DistilGPT-2"),
    # Domain models
    ("microsoft/biogpt", "BioGPT"),
    # Non english decoder
    ("asi/gpt-fr-cased-small", "GPT-fr-small"),
    ("sdadas/polish-gpt2-small", "PolishGPT-2-small"),
]

DEFAULT_DECODER_HUMAN_MODEL_NAME_SET = {
    human_model_name for _, human_model_name in DEFAULT_DECODER_MODEL_NAMES_AND_HUMAN_NAMES
}

DEFAULT_ENCODER_DECODER_MODEL_NAMES_AND_HUMAN_NAMES: List[Tuple[str, str]] = [
    # English encoder-decoder
    ("t5-small", "T5-small"),
    ("t5-base", "T5-base"),
    ("google/t5-v1_1-small", "T5-small-v1.1"),
    ("google/t5-v1_1-base", "T5-base-v1.1"),
    ("google/t5-small-lm-adapt", "T5-small-v1.1-lm-adapt"),
    ("google/t5-base-lm-adapt", "T5-base-v1.1-lm-adapt"),
    ("google/t5-efficient-tiny", "T5-efficient-tiny"),
    ("google/t5-efficient-mini", "T5-efficient-mini"),
    ("google/t5-efficient-small", "T5-efficient-small"),
    ("google/t5-efficient-base", "T5-efficient-base"),
    ("google/switch-base-8", "Switch-base-8"),
    ("google/flan-t5-small", "FLAN-T5-small"),
    ("google/flan-t5-base", "FLAN-T5-base"),
    # Multilingual encoder-decoder
    ("google/mt5-small", "mT5-small"),
    ("google/mt5-base", "mT5-base"),
    ("google/byt5-small", "ByT5-small"),
    ("google/byt5-base", "ByT5-base"),
    # Long sequence encoder-decoder
    ("google/long-t5-tglobal-base", "LongT5-TGlobal-base"),
    ("google/long-t5-local-base", "LongT5-Local-base"),
]

DEFAULT_ENCODER_DECODER_HUMAN_MODEL_NAME_SET = {
    human_model_name for _, human_model_name in DEFAULT_ENCODER_DECODER_MODEL_NAMES_AND_HUMAN_NAMES
}

DEFAULT_MODEL_NAMES_AND_HUMAN_NAMES: List[Tuple[str, str]] = (
    []
    + DEFAULT_ENCODER_MODEL_NAMES_AND_HUMAN_NAMES
    + DEFAULT_DECODER_MODEL_NAMES_AND_HUMAN_NAMES
    + DEFAULT_ENCODER_DECODER_MODEL_NAMES_AND_HUMAN_NAMES
)

DEFAULT_MODEL_NAMES = [model_name for model_name, _ in DEFAULT_MODEL_NAMES_AND_HUMAN_NAMES]

DEFAULT_MODEL_HUMAN_NAMES = [human_model_name for _, human_model_name in DEFAULT_MODEL_NAMES_AND_HUMAN_NAMES]

MAP_MODEL_NAME_TO_HUMAN_NAME = {
    model_name: human_model_name for model_name, human_model_name in DEFAULT_MODEL_NAMES_AND_HUMAN_NAMES
}

MAP_HUMAN_NAME_TO_MODEL_NAME = {
    human_model_name: model_name for model_name, human_model_name in DEFAULT_MODEL_NAMES_AND_HUMAN_NAMES
}


def is_encoder_model_name(name: str) -> bool:
    return name in DEFAULT_ENCODER_HUMAN_MODEL_NAME_SET


def is_decoder_model_name(name: str) -> bool:
    return name in DEFAULT_DECODER_HUMAN_MODEL_NAME_SET


def is_encoder_decoder_model_name(name: str) -> bool:
    return name in DEFAULT_ENCODER_DECODER_HUMAN_MODEL_NAME_SET


def filter_model_names(
    model_names: List[str], return_encoder: bool, return_decoder: bool, return_encoder_decoder: bool
) -> List[str]:
    return [
        model_name
        for model_name in model_names
        if (return_encoder and is_encoder_model_name(model_name))
        or (return_decoder and is_decoder_model_name(model_name))
        or (return_encoder_decoder and is_encoder_decoder_model_name(model_name))
    ]
