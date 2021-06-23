from pathlib import Path
from typing import Final, List

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from extinct.main import Config, start

CFG_PTH: Final[str] = "../extinct/configs"
SCHEMAS: Final[List[str]] = [
    "data=celeba_local",
    "exp=unit_test",
    "trainer=unit_test",
]


def test_entrypoint() -> None:
    """Quick run on models to check nothing's broken.

    Use this if you need an entrypoint.
    """
    with initialize(config_path=CFG_PTH):
        # config is relative to a module
        hydra_cfg = compose(
            config_name="main",
            overrides=[f"model=kc"] + SCHEMAS,
        )
        if hasattr(hydra_cfg.data, "data_dir"):
            hydra_cfg.data.data_dir = Path(hydra_cfg.data.data_dir).expanduser()
        cfg: Config = instantiate(hydra_cfg, _recursive_=True, _convert_="partial")
        start(cfg, raw_config=OmegaConf.to_container(hydra_cfg, resolve=True, enum_to_str=True))
