from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from phd_model_evaluations.data.statistics.loss_statistics import LossStatistics
from phd_model_evaluations.train.data_utils import prepare_data_loader_for_text
from phd_model_evaluations.utils.type_utils import MODEL_TYPES


def compute_loss(
    file_path: Path,
    model: MODEL_TYPES,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
    model_human_name: Optional[str] = None,
    pad_to_max_length: bool = False,
    join_examples: bool = False,
) -> LossStatistics:
    model_class_name = model.__class__.__name__
    data_loader = prepare_data_loader_for_text(
        file_path,
        model_class_name,
        tokenizer,
        batch_size,
        max_length,
        pad_to_max_length=pad_to_max_length,
        join_examples=join_examples,
    )

    loss_list = []
    for batch in tqdm(data_loader, desc="Computing loss"):
        output = model(**{k: v.to(device=device) for k, v in batch.items()})
        loss_item = output.loss.detach().item()
        del output
        loss_list.append(loss_item)

    return LossStatistics(
        model_name=model_class_name,
        model_class_name=model_class_name,
        model_human_name=model_human_name,
        join_examples=join_examples,
        sequence_length=max_length,
        loss=float(np.array(loss_list).mean()),
    )
