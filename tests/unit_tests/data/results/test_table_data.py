from phd_model_evaluations.data.results.table_data import TableData


def test_table_data() -> None:
    table_data = TableData(
        column_names=["Name", "X", "Y"],
        row_names=["simple", "advance"],
        row_data=[
            {"X": "12", "Y": "14"},
            {"X": "13", "Y": "12"},
        ],
    )
    assert table_data.column_names == ["Name", "X", "Y"], "Invalid column names"
    assert table_data.row_names == ["simple", "advance"], "Invalid row names"
