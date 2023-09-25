from typing import Any, Dict, List, Union

from transformers import (
    AlbertForMaskedLM,
    BartForConditionalGeneration,
    BertForMaskedLM,
    DistilBertForMaskedLM,
    GPT2LMHeadModel,
    GPT2Model,
    MobileBertForMaskedLM,
    OpenAIGPTLMHeadModel,
    OPTForCausalLM,
    RobertaForMaskedLM,
    RobertaModel,
    SwitchTransformersForConditionalGeneration,
    T5ForConditionalGeneration,
)

TYPE_DATASET_DICT = Dict[str, Any]
TYPE_DATASET_ELEMENT_LIST_DICT = Dict[str, List[Any]]
TYPE_DATASET_FEATURE_TO_DEFINITION_DICT = Dict[str, Dict[str, Any]]
TYPE_DATASET_LABELS = Union[List[str], List[int]]

MODEL_TYPES = Union[
    RobertaForMaskedLM,
    BertForMaskedLM,
    AlbertForMaskedLM,
    DistilBertForMaskedLM,
    MobileBertForMaskedLM,
    GPT2LMHeadModel,
    OpenAIGPTLMHeadModel,
    OPTForCausalLM,
    T5ForConditionalGeneration,
]
CLM_MODEL_TYPES = Union[GPT2LMHeadModel, OpenAIGPTLMHeadModel, OPTForCausalLM]
MLM_MODEL_TYPES = Union[
    RobertaForMaskedLM, BertForMaskedLM, AlbertForMaskedLM, DistilBertForMaskedLM, MobileBertForMaskedLM
]
SEQ2SEQ_MODEL_TYPES = Union[
    T5ForConditionalGeneration, BartForConditionalGeneration, SwitchTransformersForConditionalGeneration
]

MODEL_WITHOUT_LM_HEAD_TYPES = Union[
    RobertaModel,
    GPT2Model,
]
