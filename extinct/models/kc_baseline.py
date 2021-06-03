from typing import Dict

from kit import implements
import pytorch_lightning as pl
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torchmetrics

from extinct.datamodules.structures import DataBatch
from extinct.models.predefined import Mp64x64Net


class KCBaseline(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float, batch_norm: bool):
        super().__init__()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.net = Mp64x64Net(batch_norm=batch_norm, in_chans=3, target_dim=1)

        self.test_acc = torchmetrics.Accuracy()

    @implements(pl.LightningModule)
    def configure_optimizers(
        self,
    ) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @implements(pl.LightningModule)
    def training_step(self, batch: DataBatch, batch_idx: int) -> Tensor:
        out = self(batch.x)
        return F.binary_cross_entropy_with_logits(
            input=out, target=batch.y.float(), weight=batch.iw
        )

    @implements(pl.LightningModule)
    def test_step(self, batch: DataBatch, batch_idx: int) -> Dict[str, Tensor]:
        out = self(batch.x)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        acc = self.test_acc(out.sigmoid(), batch.y)
        self.log_dict(
            {
                f"test/loss": loss.item(),
                f"test/acc": acc,
            }
        )

        return {"y": batch.y, "s": batch.s, "preds": out.sigmoid().round().squeeze(-1)}

    @implements(pl.LightningModule)
    def test_epoch_end(self, output_results: List[Dict[str, Tensor]]) -> None:
        all_y = torch.cat([_r["y"] for _r in output_results], 0)
        all_s = torch.cat([_r["s"] for _r in output_results], 0)
        all_preds = torch.cat([_r["preds"] for _r in output_results], 0)

        dt = em.DataTuple(
            x=pd.DataFrame(
                torch.rand_like(all_s, dtype=float).detach().cpu().numpy(), columns=['x0']
            ),
            s=pd.DataFrame(all_s.detach().cpu().numpy(), columns=["s"]),
            y=pd.DataFrame(all_y.detach().cpu().numpy(), columns=["y"])
            if self.target_label == "y"
            else pd.DataFrame(all_s.detach().cpu().numpy(), columns=["s"]),
        )

        results = em.run_metrics(
            predictions=em.Prediction(hard=pd.Series(all_preds.detach().cpu().numpy())),
            actual=dt,
            metrics=[em.Accuracy(), em.RenyiCorrelation(), em.Yanovich()],
            per_sens_metrics=[em.Accuracy(), em.ProbPos(), em.TPR()],
        )

        acc = self.test_acc.compute().item()
        results_dict = {f"test_{self.target_name}_clf/pl_acc": acc}
        results_dict.update({f"test_{self.target_name}_clf/{k}": v for k, v in results.items()})

        self.log_dict(results_dict)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
