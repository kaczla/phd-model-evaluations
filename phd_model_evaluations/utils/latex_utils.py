from typing import Dict, List, Optional, Union

from phd_model_evaluations.data.arguments.latex_table_arguments import LatexTableArguments
from phd_model_evaluations.data.latex.latex_table_type import LatexTableType
from phd_model_evaluations.data.results.table_data import TableData

INTEND_SPACE = "    "


def escape_latex_text(text: str) -> str:
    return text.replace("_", r"\_")


def _get_table_begging(table_arguments: LatexTableArguments, number_columns: int) -> str:
    caption_and_label = ""
    if table_arguments.caption is not None:
        caption_and_label += r"\caption{" + table_arguments.caption + "}"
    if table_arguments.label is not None:
        caption_and_label += r"\label{" + table_arguments.label + r"}"
    if caption_and_label:
        caption_and_label += "\n"

    columns = "{| " + " | ".join(["l" for _ in range(number_columns)]) + r" |}"

    table_str = ""
    if table_arguments.table_type == LatexTableType.SIDEWAYSTABLE:
        table_str += r"\begin{sidewaystable}" + "\n"
        table_str += caption_and_label
        table_str += r"\centering" + "\n"
        table_str += r"\begin{tabular}" + columns + "\n"

    elif table_arguments.table_type == LatexTableType.LONGTABLE:
        table_str += r"\begin{longtable}" + columns + "\n"
        table_str += caption_and_label
        table_str += INTEND_SPACE + r"\\" + "\n"

    else:
        table_str += r"\begin{table}" + "\n"
        table_str += caption_and_label
        table_str += r"\centering" + "\n"
        table_str += r"\begin{tabular}" + columns + "\n"

    table_str += INTEND_SPACE + r"\hline" + "\n"

    return table_str


def _get_table_ending(table_arguments: LatexTableArguments) -> str:
    table_str = ""

    if table_arguments.table_type == LatexTableType.SIDEWAYSTABLE:
        table_str += r"\end{tabular}" + "\n"
        table_str += r"\end{sidewaystable}"

    elif table_arguments.table_type == LatexTableType.LONGTABLE:
        table_str += r"\end{longtable}"

    else:
        table_str += r"\end{tabular}" + "\n"
        table_str += r"\end{table}"

    table_str += "\n"
    return table_str


def convert_table_data_to_latex_table(table_data: TableData, table_arguments: LatexTableArguments) -> str:
    table_str = ""
    table_str += _get_table_begging(table_arguments, len(table_data.column_names))

    table_str += (
        INTEND_SPACE
        + convert_to_latex_table_row(
            table_data.column_names,
            bold=True,
            rotate=table_arguments.rotate_header_table,
            name_mapping=table_arguments.mapping_label_dict,
            mapping_parabox_size_dict=table_arguments.mapping_parabox_size_dict,
        )
        + "\n"
    )
    table_str += INTEND_SPACE + r"\hline" + "\n"

    one_line_row_names = set(table_data.one_line_row_names)
    # Skip first column = Name
    data_column_names = table_data.column_names if table_data.skip_row_name else table_data.column_names[1:]
    assert len(table_data.row_names) == len(
        table_data.row_data
    ), f"Invalid number of row name ({len(table_data.row_names)}) and row data ({len(table_data.row_data)})"
    selected_row_names = set(table_arguments.selected_row_names) if table_arguments.selected_row_names else set()
    for row_name, row_data in zip(table_data.row_names, table_data.row_data, strict=True):
        # Skip if is not selected row name
        if selected_row_names and row_name not in selected_row_names:
            continue

        table_str += INTEND_SPACE
        if row_name in one_line_row_names:
            table_str += r"\multicolumn{" + str(len(table_data.column_names)) + r"}{| l |}{"
            row_name = table_arguments.mapping_row_name_dict.get(row_name, row_name)
            table_str += _bold_latex_text(row_name)
            table_str += r"} \\" + "\n"
        else:
            row_name = table_arguments.mapping_row_name_dict.get(row_name, row_name)
            if not table_data.skip_row_name:
                table_str += _bold_latex_text(row_name) + " & "
            table_str += " & ".join([row_data.get(column_name, "-") for column_name in data_column_names])
            table_str += r" \\" + "\n"
        table_str += INTEND_SPACE + r"\hline" + "\n"

    table_str += _get_table_ending(table_arguments)

    table_str += "\n"
    return table_str


def convert_to_latex_table_row(
    column_names: List[str],
    bold: bool = False,
    rotate: bool = False,
    name_mapping: Optional[Dict[str, str]] = None,
    mapping_parabox_size_dict: Optional[Dict[int, Union[int, float]]] = None,
) -> str:
    name_mapping = {} if name_mapping is None else name_mapping
    mapping_parabox_size_dict = {} if mapping_parabox_size_dict is None else mapping_parabox_size_dict
    return (
        " & ".join(
            [
                convert_latex_text(
                    name_mapping.get(name, name),
                    bold=bold,
                    rotate=rotate,
                    parabox_size=mapping_parabox_size_dict.get(index, 0.0),
                )
                for index, name in enumerate(column_names)
            ]
        )
        + r" \\"
    )


def convert_latex_text(
    text: str, bold: bool = False, italic: bool = False, rotate: bool = False, parabox_size: Union[int, float] = 0.0
) -> str:
    text = text.replace("_", r"\_")
    formatted_text = _bold_latex_text(_italic_latex_text(text, italic=italic), bold=bold)
    if parabox_size > 0.0:
        formatted_text = _add_parabox(formatted_text, parabox_size)

    return _rotate_latex_text(formatted_text, rotate=rotate)


def _rotate_latex_text(text: str, rotate: bool = False) -> str:
    if rotate:
        return r"\rotatebox{90}{" + text + r"}"

    return text


def _add_parabox(text: str, parabox_size: Union[int, float]) -> str:
    parabox_size_str = str(parabox_size).replace(".", ",")
    return r"\parbox{" + parabox_size_str + r"cm}{" + text + r"}"


def _bold_latex_text(text: str, bold: bool = True) -> str:
    if bold:
        return r"\textbf{" + text + r"}"

    return text


def _italic_latex_text(text: str, italic: bool = True) -> str:
    if italic:
        return r"\textit{" + text + r"}"

    return text
