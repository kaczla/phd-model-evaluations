import pytest

from phd_model_evaluations.cli.dataset.create_lm_gap_challenge import parse_args


def test_parse_args_help() -> None:
    with pytest.raises(SystemExit):
        parse_args(cmd_args=["--help"])
