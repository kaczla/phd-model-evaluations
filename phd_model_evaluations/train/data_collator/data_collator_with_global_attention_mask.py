from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch


@dataclass
class DataCollatorWithGlobalAttentionMask:
    default_data_collator: Callable

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = self.default_data_collator(features)

        # Add global attention on CLS token (first token)
        global_attention_mask = torch.zeros_like(batch["input_ids"])
        global_attention_mask[:, 0] = 1
        batch["global_attention_mask"] = global_attention_mask

        return batch
