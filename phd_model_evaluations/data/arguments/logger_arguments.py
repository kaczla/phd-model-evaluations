from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from phd_model_evaluations.utils.common_utils import set_logging


@dataclass
class LoggerArguments:
    """Logging arguments."""

    verbose: bool = field(default=False, metadata={"help": "Enable debugging mode."})
    log_file: Optional[Path] = field(default=None, metadata={"help": "Logging file path."})


def set_logging_from_logger_arguments(logger_arguments: LoggerArguments) -> None:
    set_logging(
        debug=logger_arguments.verbose,
        log_file_path=logger_arguments.log_file,
    )
