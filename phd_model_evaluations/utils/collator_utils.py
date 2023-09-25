import logging
from typing import Callable, List, Optional, Type

from phd_model_evaluations.train.data_collator.data_collator_with_global_attention_mask import (
    DataCollatorWithGlobalAttentionMask,
)

LOGGER = logging.getLogger(__name__)

COLLATOR_CLASS_TYPES = List[Type[DataCollatorWithGlobalAttentionMask]]


def get_collator_class_list_from_model_name(model_name: str) -> Optional[COLLATOR_CLASS_TYPES]:
    model_name_lower = model_name.lower()
    if "longformer" in model_name_lower:
        return [DataCollatorWithGlobalAttentionMask]

    return None


def get_special_collator(collator: Callable, collator_class_list: COLLATOR_CLASS_TYPES) -> Callable:
    for collator_class in collator_class_list:
        LOGGER.info(f"Creating collator: {collator_class.__name__}")
        collator = collator_class(collator)

    return collator
