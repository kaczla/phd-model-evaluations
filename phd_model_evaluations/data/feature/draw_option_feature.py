from typing import Dict

from pydantic import BaseModel

from phd_model_evaluations.data.visualization.visualization_type import VisualizationType


class DrawOptionFeature(BaseModel):
    visualization_type: VisualizationType
    title_x: str
    title_y: str
    other_label_name: str
    map_key_to_label_name: Dict[str, str]
