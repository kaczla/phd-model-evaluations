import logging
from pathlib import Path

from phd_model_evaluations.data.arguments.latex_table_arguments import LatexTableArguments
from phd_model_evaluations.data.results.table_data import TableData
from phd_model_evaluations.utils.common_utils import get_open_fn
from phd_model_evaluations.utils.latex_utils import convert_table_data_to_latex_table

LOGGER = logging.getLogger(__name__)


def generate_latex_table(
    table_data: TableData,
    save_path: Path,
    table_arguments: LatexTableArguments,
) -> None:
    table_data_latex = convert_table_data_to_latex_table(table_data, table_arguments)

    LOGGER.info(f"Writing Latex table in: {save_path}")
    open_fn = get_open_fn(save_path.name)
    with open_fn(save_path, "wt") as f_write:
        f_write.write(table_data_latex)
