from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Final, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, MISSING, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from extinct.hydra.extinct.datamodules.configs import CelebaDataModuleConf
from extinct.hydra.extinct.models.configs import DinoModelConf, KCBaselineConf
from extinct.hydra.pytorch_lightning.trainer.configs import TrainerConf


@dataclass
class ExpConfig:
    log_offline: bool = False
    save_dir: Optional[str] = None
    seed: int = 42


@dataclass
class Config:
    """Configuration for this program.

    The values can be changed via yaml files and commandline arguments.
    """

    _target_: str = "extinct.main.Config"
    data: Any = MISSING
    exp: ExpConfig = MISSING
    exp_group: str = "Testing"
    model: Any = MISSING
    trainer: Any = MISSING


# ConfigStore enables type validation
cs = ConfigStore.instance()
cs.store(name="main_schema", node=Config)
cs.store(name="trainer_schema", node=TrainerConf, package="trainer")

DATA: Final[str] = "data"
cs.store(group=f"schema/{DATA}", name="celeba", node=CelebaDataModuleConf, package=DATA)

MODEL: Final[str] = "model"
cs.store(group=f"schema/{MODEL}", name="dino", node=DinoModelConf, package=MODEL)
cs.store(group=f"schema/{MODEL}", name="kc", node=KCBaselineConf, package=MODEL)


@hydra.main(config_path="configs", config_name="main")
def launcher(hydra_config: DictConfig) -> None:
    """Instantiate with hydra and get the experiments running!"""
    if hasattr(hydra_config.data, "data_dir"):
        hydra_config.data.data_dir = Path(hydra_config.data.data_dir).expanduser()
    cfg: Config = instantiate(hydra_config, _recursive_=True, _convert_="partial")
    start(cfg, raw_config=OmegaConf.to_container(hydra_config, resolve=True, enum_to_str=True))


def start(cfg: Config, raw_config: Optional[Dict[str, Any]]) -> None:
    """Script entrypoint."""
    pl.seed_everything(cfg.exp.seed)

    print(f"Current working directory: '{os.getcwd()}'")

    print("-----\n" + str(raw_config) + "\n-----")
    exp_logger = WandbLogger(
        entity="predictive-analytics-lab",
        project="extinct",
        offline=cfg.exp.log_offline,
        group=cfg.exp_group,
        reinit=True,
    )
    exp_logger.log_hyperparams(raw_config)
    cfg.trainer.logger = exp_logger

    cfg.data.prepare_data()
    cfg.data.setup()

    cfg.trainer.fit(model=cfg.model, datamodule=cfg.data)
    cfg.trainer.test(model=cfg.model, datamodule=cfg.data)

    exp_logger.experiment.finish()


if __name__ == '__main__':
    launcher()
