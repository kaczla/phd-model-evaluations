import logging
from typing import Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from phd_model_evaluations.data.tokenization.sub_word_type_detector import SubWordTypeDetector

LOGGER = logging.getLogger(__name__)


def get_sub_word_type_detector(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> SubWordTypeDetector:
    """
    Returns detector for identifying that given sub-token can be whole word or inner part of word.

    Example:
        - in the simple BPE approach, "</w>" is the ending characters to identifying whole words,
        - in the BPE (e.g. BERT)  approach, "##" is the beginning characters to identify inner words,
        - in the byte level BPE (e.g. GPT-2) approach, "Ġ" is the beginning characters to identify whole words,
        - in the SentencePiece (e.g. T-5) approach, "▁" is the beginning characters to identify whole words.

    Args:
        tokenizer: tokenizer to check

    Returns:
        Tuple of characters:
        - (optional) beginning characters to identifying words,
        - (optional) ending characters to identifying words.

    """
    tokenizer_name = str(tokenizer.__class__.__name__)
    if tokenizer_name.endswith("Fast"):
        tokenizer_name = tokenizer_name[:-4]

    # Maybe is the smartest way, but now it is enough
    if tokenizer_name in {"GPT2Tokenizer", "RobertaTokenizer"}:
        # Inner tokens: ing ware
        # Whole tokens: Ġsell ĠFound
        LOGGER.debug("Using sub-word detector for GPT2-base tokenizer")
        return SubWordTypeDetector(whole_word_begging="Ġ")

    elif tokenizer_name in {"BertTokenizer"}:
        # Inner tokens: ##tier ##ing
        # Whole tokens: buy This
        LOGGER.debug("Using sub-word detector for BERT-base tokenizer")
        return SubWordTypeDetector(inner_word_begging="##")

    elif tokenizer_name in {"T5Tokenizer"}:
        # Inner tokens: scribe ives
        # Whole tokens: ▁height ▁Perhaps
        LOGGER.debug("Using sub-word detector for SentencePiece")
        return SubWordTypeDetector(whole_word_begging="▁")

    LOGGER.warning(f"Cannot found sub-word detector for tokenizer: {tokenizer.__class__.__name__}")
    return SubWordTypeDetector()


def clean_token(token: str) -> str:
    return token.replace("\t", " ").strip()
