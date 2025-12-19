import yaml
from pathlib import Path
from importlib.resources import files


def get_default_tdmpc2_config():
    package_path = Path(str(files("layeredrl")))
    tdmpc2_config_path = package_path / "tdmpc2" / "default_cfg.yaml"
    with open(tdmpc2_config_path, "r") as file:
        tdmpc2_config = yaml.safe_load(file)
    return tdmpc2_config
