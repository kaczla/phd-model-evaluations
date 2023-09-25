import logging
from csv import DictReader
from io import StringIO
from pathlib import Path
from sys import stdin
from typing import Dict, List

from phd_model_evaluations.utils.common_utils import get_open_fn

LOGGER = logging.getLogger(__name__)


def get_input_text_from_stdio() -> List[str]:
    return stdin.readlines()


def get_input_text(file_path: Path) -> List[str]:
    if str(file_path) == "-":
        LOGGER.info("Reading text from STDIN")
        text = get_input_text_from_stdio()
        return text

    if not file_path.exists():
        raise RuntimeError(f"File does not exist: {file_path}")

    LOGGER.info(f"Reading text from: {file_path}")
    open_fn = get_open_fn(file_path.name)
    with open_fn(file_path, "rt") as f_read:
        text = f_read.readlines()

    return text


def convert_tsv_to_json(text: List[str]) -> List[Dict]:
    if len(text) == 1:
        return [{"data": text[0].strip()}]

    reader = DictReader(StringIO("".join(text)), delimiter="\t")
    return [convert_data(data) for data in reader]


def convert_data(data: Dict[str, str]) -> Dict[str, str]:
    new_data: Dict[str, str] = {}
    for key, value in data.items():
        if value == "NaN":
            value = ""
        new_data[key] = value
    return new_data
