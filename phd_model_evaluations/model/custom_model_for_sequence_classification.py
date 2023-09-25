import logging
from typing import Any, Optional, Tuple, Type, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from phd_model_evaluations.model.custom_classifier import CustomClassifier
from phd_model_evaluations.utils.type_utils import MODEL_WITHOUT_LM_HEAD_TYPES

LOGGER = logging.getLogger(__name__)


class CustomModelForSequenceClassification(PreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        model_cls: Type[MODEL_WITHOUT_LM_HEAD_TYPES],
        model: Optional[MODEL_WITHOUT_LM_HEAD_TYPES] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        if model is not None:
            LOGGER.info(f"Loading base model from weights: {model.__class__.__name__}")
            self.model = model
        else:
            LOGGER.info(f"Loading base model from class: {model_cls.__name__}")
            self.model = model_cls(config)
        self.classifier = CustomClassifier(config)

    def forward(  # noqa: C901
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        if not self.classifier.has_cls_token:
            batch_size, sequence_length = input_ids.shape[:2]  # type: ignore[union-attr]
            assert (
                self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."
            sequence_lengths: Union[float, torch.Tensor]
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
                else:
                    sequence_lengths = -1
                    LOGGER.warning(
                        f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                        "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                    )
            logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            loss_fct: Union[torch.nn.MSELoss, torch.nn.CrossEntropyLoss, torch.nn.BCEWithLogitsLoss]
            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _init_weights(self, _: Any) -> None:
        return

    def resize_position_embeddings(self, new_num_position_embeddings: int) -> None:
        self.model.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Union[torch.nn.Embedding, Tuple[torch.nn.Embedding]]:
        position_embeddings: Union[torch.nn.Embedding, Tuple[torch.nn.Embedding]] = self.model.get_position_embeddings()
        return position_embeddings

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past: Any, beam_idx: Any) -> Any:
        return self.model._reorder_cache(past, beam_idx)
