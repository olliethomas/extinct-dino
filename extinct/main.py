import copy
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

from extinct.hydra.extinct.datamodules.configs import (  # type: ignore[import]
    CelebaDataModuleConf,
)
from extinct.hydra.extinct.models.configs import (  # type: ignore[import]
    CelebaErmBaselineConf,
    CelebaKCBaselineConf,
    LaftrBaselineConf,
)
from extinct.hydra.extinct.models.dino.configs import DINOConf  # type: ignore[import]
from extinct.hydra.pytorch_lightning.trainer.configs import (  # type: ignore[import]
    TrainerConf,
)
from extinct.utils.callbacks import IterationBasedProgBar


@dataclass
class ExpConfig:
    es_patience: int = 3
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
cs.store(group=f"schema/{MODEL}", name="kc", node=CelebaKCBaselineConf, package=MODEL)
cs.store(group=f"schema/{MODEL}", name="erm", node=CelebaErmBaselineConf, package=MODEL)
cs.store(group=f"schema/{MODEL}", name="laftr", node=LaftrBaselineConf, package=MODEL)
cs.store(group=f"schema/{MODEL}", name="dino", node=DINOConf, package=MODEL)


@hydra.main(config_path="configs", config_name="main")
def launcher(hydra_config: DictConfig) -> None:
    """Instantiate with hydra and get the experiments running!"""
    if hasattr(hydra_config.data, "data_dir"):
        hydra_config.data.data_dir = Path(hydra_config.data.data_dir).expanduser()
    cfg: Config = instantiate(hydra_config, _recursive_=True, _convert_="partial")
    start(cfg, raw_config=OmegaConf.to_container(hydra_config, resolve=True, enum_to_str=True))


def start(cfg: Config, raw_config: Optional[Dict[str, Any]]) -> None:
    """Script entrypoint."""
    print(f"Current working directory: '{os.getcwd()}'")
    print("-----\n" + str(raw_config) + "\n-----")

    exp_logger = WandbLogger(
        entity="predictive-analytics-lab",
        project="extinct",
        offline=cfg.exp.log_offline,
        group=cfg.exp_group,
        reinit=True,  # for multirun compatibility
    )

    exp_logger.log_hyperparams(raw_config)
    cfg.trainer.logger = exp_logger

    pl.seed_everything(cfg.exp.seed)
    cfg.data.prepare_data()
    cfg.data.setup()

    cfg.model.target = cfg.data.train_data.dataset.dataset.ti.y_label
    callbacks: list[pl.Callback] = [IterationBasedProgBar()]
    cfg.trainer.callbacks = callbacks

    # Build the model
    cfg.model.build(datamodule=copy.deepcopy(cfg.data), trainer=copy.deepcopy(cfg.trainer))
    # Fit the model
    cfg.trainer.fit(model=cfg.model, datamodule=cfg.data)
    # Test the model
    cfg.trainer.test(model=cfg.model, datamodule=cfg.data)
    # Test the model with additional target attributes
    cfg.data.train_data.dataset.dataset.ti.new_task("Smiling")  # Amends the underlying dataset
    cfg.model.target = cfg.data.train_data.dataset.dataset.ti.y_label
    cfg.trainer.test(model=cfg.model, datamodule=cfg.data)
    cfg.data.train_data.dataset.dataset.ti.new_task("Rosy_Cheeks")  # Amends the underlying dataset
    cfg.model.target = cfg.data.train_data.dataset.dataset.ti.y_label
    cfg.trainer.test(model=cfg.model, datamodule=cfg.data)

    exp_logger.experiment.finish()


if __name__ == "__main__":
    launcher()
