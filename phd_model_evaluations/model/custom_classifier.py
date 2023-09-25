from typing import Any

import torch
from transformers import PretrainedConfig


class CustomClassifier(torch.nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.has_cls_token = config.has_cls_token
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.classifier_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor, **_: Any) -> torch.Tensor:
        # take <s> token (equivalent to [CLS]) or last hidden state
        x = features[:, 0, :] if self.has_cls_token else features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
