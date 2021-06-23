from typing import Sequence, Tuple, Union

from pytorch_lightning import Callback, LightningModule, Trainer
import torch
from torch import Tensor
from torch.optim import Optimizer


class SSLOnlineEvaluator(Callback):
    """
    Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    Your model should have:
        - ``self.student_network``
        - ``self.teacher_network``
        - ``self.datamodule``
        - ``self.lin_clf_epochs``
        - ``self.lr_eval``
        - ``self.target``

    Example::
        # your model must have 6 attributes
        model = Model()
        model.student = ... #
        model.datamodule = ... #
        model.lin_clf_epochs = ... #
        model.lr_eval = ... #
        model.target = ... #

        and, if using KNN classifier:
        model._encode_dataset = ... #

        online_eval = SSLOnlineEvaluator()
    """

    def __init__(self):
        super().__init__()
        self.optimizer: Optimizer

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
