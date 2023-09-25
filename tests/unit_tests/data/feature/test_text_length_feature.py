from phd_model_evaluations.data.feature.aggregated_feature import AggregatedFeature
from phd_model_evaluations.data.feature.checker.text_length_feature import TextLengthFeature
from phd_model_evaluations.data.feature.draw_option_feature import DrawOptionFeature
from phd_model_evaluations.data.statistics.score_statistics import ScoreStatistics
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType


def test_text_length_feature() -> None:
    feature = TextLengthFeature(text_length=15)
    assert feature.name == "text_length", "Invalid feature name"
    assert feature.text_length == 15, "Invalid text length in the feature"
    assert feature.get_aggregation_data() == {15: 1}, "Invalid aggregation data from the feature"


def test_text_length_feature_aggregation() -> None:
    aggregation = TextLengthFeature.aggregate(
        [
            TextLengthFeature(text_length=15),
            TextLengthFeature(text_length=12),
            TextLengthFeature(text_length=20),
            TextLengthFeature(text_length=15),
            TextLengthFeature(text_length=13),
        ]
    )
    expected_aggregation = AggregatedFeature(
        name="text_length",
        data={15: 2, 12: 1, 20: 1, 13: 1},
        total=5,
        statistics=ScoreStatistics(
            total_elements=5, max_value=20.0, min_value=12.0, avg_value=15.0, std_value=2.756809750418044
        ),
        draw_option=DrawOptionFeature(
            visualization_type=VisualizationType.histogram,
            title_x="Długość tekstu",
            title_y="Liczba wystąpień",
            other_label_name="-",
            map_key_to_label_name={},
        ),
    )
    assert aggregation == expected_aggregation, "Invalid aggregated features"
    assert aggregation.dict() == {
        "name": "text_length",
        "data": {15: 2, 12: 1, 20: 1, 13: 1},
        "total": 5,
        "statistics": {
            "total_elements": 5,
            "max_value": 20.0,
            "min_value": 12.0,
            "avg_value": 15.0,
            "std_value": 2.756809750418044,
        },
        "draw_option": {
            "visualization_type": "histogram",
            "title_x": "Długość tekstu",
            "title_y": "Liczba wystąpień",
            "other_label_name": "-",
            "map_key_to_label_name": {},
        },
    }, "Invalid aggregated features data"
