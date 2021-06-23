from typing import Optional, Sequence, Tuple, Union

from pytorch_lightning import Callback, LightningModule, Trainer
import torch
from torch import Tensor
from torch.optim import Optimizer


class SSLOnlineEvaluator(Callback):
    """
    Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    Example::
        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
            dataset='imagenet'
        )
    """

    def __init__(
        self,
        dataset: str = 'akdsfh',
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        z_dim: int = None,
        num_classes: int = None,
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer: Optimizer

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset = dataset

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from extinct.components.models.dino import DINOLinearClassifier

        pl_module.eval_clf = DINOLinearClassifier(
            enc=pl_module.student.backbone,
            target_dim=pl_module.datamodule.y_dim,
            epochs=pl_module.lin_clf_epochs,
            weight_decay=0,
            lr=pl_module.lr_eval,
        ).to(pl_module.device)
        pl_module.eval_clf.target = pl_module.target

        self.optimizer = torch.optim.Adam(pl_module.eval_clf.parameters(), lr=1e-4)

    def to_device(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # get the labeled batch
        inputs, s, y = batch

        # last input is for online eval
        x = inputs.to(device)
        s = s.to(device)
        y = y.to(device)

        return x, s, y

    def _eval_loop(self, pl_module: LightningModule):
        from extinct.components.models.dino import EvalMethod

        if pl_module.eval_method is EvalMethod.lin_clf:
            pl_module.eval_trainer.fit(
                pl_module.eval_clf,
                train_dataloader=pl_module.datamodule.train_dataloader(
                    eval=True, batch_size=pl_module.batch_size_eval
                ),
            )
        else:
            from extinct.components.models.dino import KNN

            train_data_encoded = pl_module._encode_dataset(stage="train")
            self.eval_clf = KNN(
                train_features=train_data_encoded.x, train_labels=train_data_encoded.y
            )
            self.eval_clf.target = pl_module.target
            self.eval_clf.to(pl_module.device)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self._eval_loop(pl_module)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._eval_loop(pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._eval_loop(pl_module)
