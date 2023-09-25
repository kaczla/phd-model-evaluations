import pytest

from phd_model_evaluations.cli.output.convert_tsv_to_json import parse_args


def test_parse_args_help() -> None:
    with pytest.raises(SystemExit):
        parse_args(cmd_args=["--help"])
