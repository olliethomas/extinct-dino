from typing import Sequence

from pytorch_lightning import Callback, LightningModule, Trainer
from torch.nn import Module

from extinct.utils import cosine_scheduler


class DINOMAWeightUpdate(Callback):
    """
    Weight update rule from BYOL.
    Your model should have:
        - ``self.student_network``
        - ``self.teacher_network``
    Updates the teacher_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.student_network = ...
        model.teacher_network = ...
        trainer = Trainer(callbacks=[DINOMAWeightUpdate()])
    """

    def __init__(self, max_steps: int, initial_tau: float = 0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()

        self.momentum_schedule = cosine_scheduler(
            base_value=initial_tau,
            final_value=1,
            total_iters=max_steps,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # get networks
        student_net = pl_module.student
        teacher_net = pl_module.teacher

        # update weights
        self.update_weights(pl_module.global_step, student_net, teacher_net)

    def update_weights(self, train_itrs: int, student: Module, teacher: Module) -> None:
        # apply MA weight update
        em = self.momentum_schedule[train_itrs]  # momentum parameter
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
