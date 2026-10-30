import argparse

import pytest
import yaml

from src.config import ConfigParser, str2bool


@pytest.mark.parametrize("value, expected", [
    ("True", True), ("true", True), ("1", True), ("yes", True),
    ("False", False), ("false", False), ("0", False), ("no", False),
])
def test_str2bool(value, expected):
    assert str2bool(value) is expected


def test_str2bool_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")


@pytest.fixture
def config_files(tmp_path):
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text(yaml.safe_dump({
        "debug": True,
        "training": {"epochs": 1, "batch_size": 4},
    }))
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump({"training": {"epochs": 500}}))
    return defaults, experiment


def test_experiment_config_is_merged_over_defaults(config_files, monkeypatch):
    defaults, experiment = config_files
    monkeypatch.setattr("sys.argv", ["main.py", "--config", str(experiment)])

    config = ConfigParser(default_config_path=str(defaults)).parse()

    assert config["training"] == {"epochs": 500, "batch_size": 4}


def test_debug_false_flag_disables_debug(config_files, monkeypatch):
    defaults, _ = config_files
    monkeypatch.setattr("sys.argv", ["main.py", "--debug", "False"])

    config = ConfigParser(default_config_path=str(defaults)).parse()

    assert config["debug"] is False
