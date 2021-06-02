from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, Final

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, MISSING
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from extinct.hydra.extinct.models.configs import DinoModelConf
from extinct.hydra.fair_bolts.datamodules.configs import AdultDataModuleConf, CelebaDataModuleConf
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
cs.store(group=f"schema/{DATA}", name="adult", node=AdultDataModuleConf, package=DATA)

MODEL: Final[str] = "model"
cs.store(group=f"schema/{MODEL}", name="dino", node=DinoModelConf, package=MODEL)


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

    exp_logger = WandbLogger(
        entity="predictive-analytics-lab",
        project="extinct",
        offline=cfg.exp.log_offline,
        group= cfg.exp_group,
    )
    exp_logger.log_hyperparams(raw_config)
    cfg.trainer.logger = exp_logger

    cfg.trainer.fit(model=cfg.model, datamodule=cfg.data)
    cfg.trainer.test(model=cfg.model, datamodule=cfg.data)

    exp_logger.experiment.finish()


if __name__ == '__main__':
    launcher()