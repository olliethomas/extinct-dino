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
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from extinct.hydra.extinct.datamodules.configs import CelebaDataModuleConf
from extinct.hydra.extinct.models.configs import (
    DinoModelConf,
    ErmBaselineConf,
    KCBaselineConf,
    LaftrBaselineConf,
)
from extinct.hydra.pytorch_lightning.trainer.configs import (
    TrainerConf,  # type: ignore[import]
)
from extinct.models.aux_classifier import AuxClassifier
from extinct.utils.callbacks import IterationBasedProgBar


@dataclass
class ExpConfig:
    early_stopping: bool = True
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
    aux_trainer: Any = MISSING


# ConfigStore enables type validation
cs = ConfigStore.instance()
cs.store(name="main_schema", node=Config)
cs.store(name="trainer_schema", node=TrainerConf, package="trainer")
cs.store(name="aux_trainer_schema", node=TrainerConf, package="aux_trainer")

DATA: Final[str] = "data"
cs.store(group=f"schema/{DATA}", name="celeba", node=CelebaDataModuleConf, package=DATA)

MODEL: Final[str] = "model"
cs.store(group=f"schema/{MODEL}", name="dino", node=DinoModelConf, package=MODEL)
cs.store(group=f"schema/{MODEL}", name="kc", node=KCBaselineConf, package=MODEL)
cs.store(group=f"schema/{MODEL}", name="erm", node=ErmBaselineConf, package=MODEL)
cs.store(group=f"schema/{MODEL}", name="laftr", node=LaftrBaselineConf, package=MODEL)


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
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.00,
        patience=cfg.exp.es_patience,
        verbose=False,
    )
    if cfg.exp.early_stopping:
        cfg.trainer.callbacks += [early_stop_callback]

    pl.seed_everything(cfg.exp.seed)
    cfg.data.prepare_data()
    cfg.data.setup()

    cfg.model.target = cfg.data.train_data.dataset.dataset.ti.y_label
    cfg.trainer.callbacks = [IterationBasedProgBar()]

    cfg.model.target = cfg.data.train_data.dataset.dataset.ti.y_label
    cfg.trainer.fit(cfg.model, datamodule=cfg.data)
    fit_and_test(cfg)

    for additional_target in ("Smiling", "Rosy_Cheeks"):
        cfg.data.train_data.dataset.dataset.ti.new_task(
            additional_target
        )  # Amends the underlying dataset
        cfg.model.target = cfg.data.train_data.dataset.dataset.ti.y_label
        fit_and_test(cfg)

    # Manually invoke finish for multirun-compatibility
    exp_logger.experiment.finish()


def fit_and_test(cfg):
    _trainer = copy.deepcopy(cfg.aux_trainer)
    clf_model = AuxClassifier(
        enc=cfg.model.enc, classifier=cfg.model.clf, lr=1e-3, weight_decay=1e-8, lr_gamma=0.999
    )
    clf_model.target = cfg.data.train_data.dataset.dataset.ti.y_label
    _trainer.fit(clf_model, datamodule=cfg.data)
    _trainer.test(clf_model, datamodule=cfg.data)


if __name__ == "__main__":
    launcher()
