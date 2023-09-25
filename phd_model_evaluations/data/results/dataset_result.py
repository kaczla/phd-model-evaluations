from phd_model_evaluations.data.results.empty_dataset_result import EmptyDatasetResult


class DatasetResult(EmptyDatasetResult):
    score: float

    def get_score(self, score_factor: float = 1.0) -> float:
        return self.score * score_factor
