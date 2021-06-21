from pathlib import Path
from typing import Final, List

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from extinct.main import Config, start

CFG_PTH: Final[str] = "../extinct/configs"
SCHEMAS: Final[List[str]] = [
    "data=celeba_local",
    "exp=unit_test",
    "trainer=unit_test",
    "aux_trainer=unit_test",
]


@pytest.mark.parametrize("model_schema", ["erm", "kc", "laftr", "local_dino"])
def test_entrypoint(model_schema: str) -> None:
    """Quick run on models to check nothing's broken.

    Use this if you need an entrypoint.
    """
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="main",
            overrides=[f"model={model_schema}"] + SCHEMAS,
        )
        if hasattr(hydra_cfg.data, "data_dir"):
            hydra_cfg.data.data_dir = Path(hydra_cfg.data.data_dir).expanduser()
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        start(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))


@pytest.mark.parametrize("fairness", ["DP", "EO", "EqOp"])
def test_laftr_variants(fairness: str) -> None:
    """Quick run on models to check nothing's broken.

    Use this if you need an entrypoint.
    """
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="main",
            overrides=[f"model=laftr", f"model.fairness={fairness}"] + SCHEMAS,
        )
        if hasattr(hydra_cfg.data, "data_dir"):
            hydra_cfg.data.data_dir = Path(hydra_cfg.data.data_dir).expanduser()
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        start(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))
