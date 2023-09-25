from typing import List, Tuple

from phd_model_evaluations.data.model.task_specific_paramaters import TaskSpecificParameters


class Seq2SeqTaskSpecificParameters(TaskSpecificParameters):
    prefix_and_key_text: List[Tuple[str, str]]
    target_length: int
    dynamic_generation_length: bool
    generation_max_length: int
    generation_num_beams: int
