import json
import logging
from gzip import open as open_gz
from lzma import open as open_lzma
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TypeVar, Union

import numpy as np

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T_DICT = TypeVar("T_DICT", bound=Dict[str, Union[str, int, float]])


LOGGER = logging.getLogger(__name__)


def round_float(value: float, precision: Optional[int] = None) -> float:
    if precision is None:
        return value

    rounded: float = np.around(value, decimals=precision).item()
    return rounded


def batchify(data: List[T], batch_size: int) -> Generator[List[T], None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def get_open_fn(file_name: str) -> Callable:
    if file_name.endswith(".xz"):
        return open_lzma

    elif file_name.endswith(".gz"):
        return open_gz

    return open


def merge_dictionaries(dictionaries: List[Dict[T, int]], min_frequency_items: int = 1) -> Dict[T, int]:
    LOGGER.debug("Merging dictionaries data ...")
    merged_dictionary = dictionaries.pop(0)
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if key in merged_dictionary:
                merged_dictionary[key] += value
            else:
                merged_dictionary[key] = value

    if min_frequency_items > 1:
        merged_dictionary = remove_low_frequency_elements_from_dictionary(merged_dictionary, min_frequency_items)

    LOGGER.debug(f"Merged {len(dictionaries)} dictionaries data")
    return merged_dictionary


def remove_low_frequency_elements_from_dictionary(dictionary: Dict[T, int], min_frequency: int) -> Dict[T, int]:
    LOGGER.debug(f"Removing low frequency items, removing elements with value below: {min_frequency}")

    key_to_removes = []
    for key, value in dictionary.items():
        if value < min_frequency:
            key_to_removes.append(key)

    for key in key_to_removes:
        del dictionary[key]

    return dictionary


def get_key_from_directory_items(dictionary_items: Tuple[T1, T2]) -> T1:
    return dictionary_items[0]


def get_value_from_directory_items(dictionary_items: Tuple[T1, T2]) -> T2:
    return dictionary_items[1]


def load_json_data(file_path: Path) -> Dict:
    LOGGER.info(f"Reading json data from: {file_path}")
    open_fn = get_open_fn(file_path.name)
    with open_fn(file_path, "rt") as f_read:
        data: Dict = json.load(f_read)
        return data


def save_json_data(
    data: Union[Dict, List[Dict], List[List[Any]]], file_path: Path, indent: int = 4, ensure_ascii: bool = False
) -> None:
    data_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

    if str(file_path) == "-":
        LOGGER.info("Writing json data to STDOUT")
        print(data_str)  # noqa: T201
        return

    LOGGER.info(f"Writing json data to: {file_path}")
    open_fn = get_open_fn(file_path.name)
    with open_fn(file_path, "wt") as f_write:
        f_write.write(data_str)


def sort_json_data(data: List[T_DICT], key_name: str) -> List[T_DICT]:
    LOGGER.info(f'Sorting JSON data with key: "{key_name}"')
    try:
        sorted_data = sorted(data, key=lambda x: x[key_name])
    except KeyError:
        LOGGER.error(f'Cannot sort list of dictionaries, missing key: "{key_name}"')
        return data

    return sorted_data


def set_logging(debug: bool = False, log_file_path: Optional[Path] = None) -> None:
    logger_handlers: Optional[List[logging.Handler]] = None

    if log_file_path is not None:
        logger_handlers = [logging.StreamHandler(), logging.FileHandler(log_file_path)]

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s",
        handlers=logger_handlers,
    )
    if debug:
        loger = logging.getLogger("urllib3")
        loger.setLevel(logging.INFO)
