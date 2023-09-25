from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.data.results.table_data import TableData


def test_aggregated_results_methods() -> None:
    aggregated_results = AggregatedResults(
        model_list=["RoBERTa-base", "BERT-base", "RoBERTa-large"],
        metrics={"CoLA": "Matthews correlation", "QQP": "F1 score", "QNLI": "Accuracy", "RTE": "Accuracy"},
        results={
            "CoLA": {"BERT-base": 60.2, "RoBERTa-base": 63.6, "RoBERTa-large": 68.0},
            "QNLI": {"BERT-base": 91.3, "RoBERTa-base": 92.8, "RoBERTa-large": 94.7},
            "QQP": {"BASE-base": 91.3, "RoBERTa-base": 91.9, "RoBERTa-large": 92.2},
            "RTE": {"BERT-base": 77.7, "RoBERTa-base": 78.7, "RoBERTa-large": 86.6},
            "LM-GAP": {"BERT-base": 756.6, "RoBERTa-base": 114.2, "RoBERTa-large": 86.1},
        },
    )
    expected_table_data = TableData(
        column_names=["LM-GAP", "CoLA", "QQP", "QNLI", "RTE"],
        row_names=[
            "RoBERTa-base",
            "RoBERTa-base data",
            "BERT-base",
            "BERT-base data",
            "RoBERTa-large",
            "RoBERTa-large data",
        ],
        row_data=[
            {},
            {"LM-GAP": "114.20", "CoLA": "63.60", "QQP": "91.90", "QNLI": "92.80", "RTE": "78.70"},
            {},
            {"LM-GAP": "756.60", "CoLA": "60.20", "QNLI": "91.30", "RTE": "77.70"},
            {},
            {"LM-GAP": "86.10", "CoLA": "68.00", "QQP": "92.20", "QNLI": "94.70", "RTE": "86.60"},
        ],
        one_line_row_names=["RoBERTa-base", "BERT-base", "RoBERTa-large"],
        skip_row_name=True,
    )

    assert aggregated_results.get_data_set_names() == ["LM-GAP", "CoLA", "QQP", "QNLI", "RTE"], "Invalid data set names"
    assert aggregated_results.get_table_data() == expected_table_data, "Invalid table data"
