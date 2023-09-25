from typing import Dict, List

from pydantic import BaseModel

from phd_model_evaluations.data.model.model_parameter_label import ModelParameterLabel


class ModelInfo(BaseModel):
    name: str
    human_name: str

    def get_table_row(self) -> Dict[str, str]:
        return {
            str(ModelParameterLabel.human_name): self.human_name,
            str(ModelParameterLabel.name): self.get_model_name_with_url(self.name),
        }

    @staticmethod
    def get_table_header() -> List[str]:
        return [str(ModelParameterLabel.human_name), str(ModelParameterLabel.name)]

    @staticmethod
    def get_model_name_with_url(model_name: str) -> str:
        return r"\href{https://huggingface.co/" + model_name + "}{" + model_name.replace("_", r"\_") + "}"
