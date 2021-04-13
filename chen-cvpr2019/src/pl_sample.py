import copy
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from kudo_model import KudoModel
from datamodule import MPIIDataModule


class PoseNet(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        gan_accuracy_cap: float = 0.9,
        heuristic_loss_weight: float = 0.5,  # XXX: source ?
        use_heuristic_loss: bool = False,
        use_sh_detection: bool = False,
        n_in: int = 34,
        n_unit: int = 1024,
        mode: str = "unsupervised",
        use_bn: bool = True,
        activate_func=F.leaky_relu,
    ):
        super().__init__()
        self.save_hyperparameters()

        # set generator and parameter
        gen_hparams = copy.deepcopy(self.hparams)
        gen_hparams["mode"] = "generator"
        self.gen = KudoModel(**gen_hparams)
        dis_hparams = copy.deepcopy(self.hparams)
        dis_hparams["mode"] = "discriminator"
        self.dis = KudoModel(**dis_hparams)

        self.train_metrics = nn.ModuleDict(
            {"discriminator_accuracy": pl.metrics.Accuracy()}
        )
        # self.val_metrics = nn.ModuleDict({"acc": pl.metrics.Accuracy()})
        # self.test_metrics = nn.ModuleDict({""})

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx):
        # xy_real: joint_num * 2(xy axis)
        # xy_proj, xyz, scale = batch
        xy_proj, xyz, scale = batch
        batch_size = len(xy_proj)
        xy_real, xyz = xy_proj[:, 0], xyz[:, 0]
        z_real = xyz.narrow(-1, 2, 1)
        z_pred = self(xy_real)
        z_mse = F.mse_loss(z_pred, z_real)
        if self.hparams.mode == "supervised":
            loss = z_mse
        elif self.hparams.mode == "unsupervised":
            # Random rotation
            # TODO: [0, 2pi) の一様分布をautogradありで生成する方法のベストプラクティスを探す
            theta = torch.rand(batch_size, 1) * 2 * np.pi
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            # 2D Projection
            # x = xy_real[:, 0::2]
            # y = xy_real[:, 1::2]

            xy_real = xy_real.view(batch_size, 17, 2)
            x = xy_real.narrow(-1, 0, 1).squeeze()
            y = xy_real.narrow(-1, 1, 1).squeeze()
            # y = xy_real[:, 1::2]
            new_x = x * cos_theta + z_pred * sin_theta
            xy_fake = torch.stack((new_x, y), dim=2)# .view(batch_size, -1)

            y_real = self.dis(xy_real)
            y_fake = self.dis(xy_fake)

            print('**** unko ***', y_real, y_real.shape)
            acc_dis_fake = pl.metrics.functional.accuracy(
                y_fake, torch.zeros_like(y_fake, dtype=torch.int)
            )
            acc_dis_real = pl.metrics.functional.accuracy(
                y_real, torch.ones_like(y_real, dtype=torch.int)
            )
            acc_dis = (acc_dis_fake + acc_dis_real) / 2

            loss_gen = F.softplus(-y_fake).sum() / batch_size
            if self.hparams.use_heuristic_loss:
                loss_heuristic = self.calculate_heuristic_loss(
                    xy_real=xy_real, z_pred=z_pred
                )
                loss_gen += loss_heuristic * self.hparams.heuristic_loss_weight
                self.log("loss_heuristic", loss_heuristic)

            loss_dis = F.softplus(-y_real).sum() / batch_size
            loss_dis += F.softplus(y_fake).sum() / batch_size

            self.log("loss_gen", loss_gen)
            self.log("z_mse", z_mse)
            self.log("loss_dis", loss_dis)
            self.log("acc_dis", acc_dis)
            self.log("acc/fake", acc_dis_fake)
            self.log("acc/real", acc_dis_real)

            if acc_dis >= (1 - self.hparams.gan_accuracy_cap):
                return loss_gen
            else:
                return loss_dis

        return loss

    def validation_step(self, batch, batch_idx):
        xy_proj, xyz, scale = batch
        xy_real, xyz = xy_proj[:, 0], xyz[:, 0]
        z_pred = self(xy_real)
        loss = F.mse_loss(z_pred, xyz.narrow(-1, 2, 1))
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        xy_proj, xyz, scale = batch
        xy_real, xyz = xy_proj[:, 0], xyz[:, 0]
        z_pred = self(xy_real)
        loss = F.mse_loss(z_pred, xyz.narrow(-1, 2, 1))
        self.log("test_loss", loss)
        # x, y = batch
        # y_hat = self(x)
        # loss = F.mse_loss(y_hat, y)
        # self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def calculate_rotation(xy_real, z_pred):
        # xy_real: batch_num * joint_num * 2(xy axis)
        xy_split = torch.split(xy_real, xy_real.shape[1], dim=1)
        z_split = torch.split(z_pred, z_pred.shape[1], dim=1)
        # Vector v0 (neck -> nose) on zx-plain. v0=(a0, b0).
        a0 = z_split[9] - z_split[8]
        b0 = xy_split[9 * 2] - xy_split[8 * 2]
        n0 = F.sqrt(a0 * a0 + b0 * b0)
        # Vector v1 (right shoulder -> left shoulder) on zx-plain. v1=(a1, b1).
        a1 = z_split[14] - z_split[11]
        b1 = xy_split[14 * 2] - xy_split[11 * 2]
        n1 = F.sqrt(a1 * a1 + b1 * b1)
        # Return sine value of the angle between v0 and v1.
        return (a0 * b1 - a1 * b0) / (n0 * n1)

    @staticmethod
    def calculate_heuristic_loss(self, xy_real, z_pred):
        return torch.mean(F.relu(-self.calculate_rotation(xy_real, z_pred)))


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MPIIDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    dm = MPIIDataModule(use_sh_detection=args.use_sh_detection)

    model = PoseNet()

    mlf_logger = MLFlowLogger(
        experiment_name="chen-cvpr2019", tracking_uri="file:./ml-runs"
    )
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=mlf_logger, auto_lr_find=True)
    trainer.tune(model, datamodule=dm)
    trainer.fit(
        model
    )

    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == "__main__":
    cli_main()
