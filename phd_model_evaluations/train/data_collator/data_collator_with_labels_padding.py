from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import DataCollatorWithPadding


@dataclass
class DataCollatorWithLabelsPadding:
    LABEL_PADDING_ID = -100
    default_data_collator: DataCollatorWithPadding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max((len(feature["labels"]) for feature in features))
        for feature in features:
            labels = feature["labels"]
            if len(labels) < max_length:
                feature["labels"] = labels + (max_length - len(labels)) * [-100]
        batch: Dict[str, Any] = self.default_data_collator(features)
        return batch
