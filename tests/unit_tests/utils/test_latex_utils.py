from phd_model_evaluations.data.arguments.latex_table_arguments import LatexTableArguments
from phd_model_evaluations.data.results.table_data import TableData
from phd_model_evaluations.utils.latex_utils import convert_table_data_to_latex_table

_EXPECTED_TABLE_STR = r"""\begin{table}
\caption{Some table}
\centering
\begin{tabular}{| l | l | l |}
    \hline
    \textbf{Name} & \textbf{X} & \textbf{Y} \\
    \hline
    \multicolumn{3}{| l |}{\textbf{Simple method}} \\
    \hline
    \textbf{simple} & 10 & 9 \\
    \hline
    \textbf{advance} & 13 & 13 \\
    \hline
\end{tabular}
\end{table}

"""


def test_get_latex_table_str() -> None:
    table_data = TableData(
        column_names=["Name", "X", "Y"],
        row_names=["Simple method", "simple", "advance"],
        row_data=[{}, {"X": "10", "Y": "9"}, {"X": "13", "Y": "13"}],
        one_line_row_names=["Simple method"],
    )

    assert (
        convert_table_data_to_latex_table(table_data, LatexTableArguments(caption="Some table")) == _EXPECTED_TABLE_STR
    ), "Invalid generated Latex table"
