from typing import List, Optional

from pydantic import BaseModel

from phd_model_evaluations.data.statistics.correlations.correlation_data import CorrelationData


class ComputedCorrelations(BaseModel):
    x_name: str
    y_name: str
    pearson_correlation: CorrelationData
    spearman_correlation: CorrelationData
    x_higher_better: bool
    y_higher_better: bool
    previous_pearson_correlation: Optional[CorrelationData] = None
    previous_spearman_correlation: Optional[CorrelationData] = None
    x_values: Optional[List[float]] = None
    y_values: Optional[List[float]] = None
