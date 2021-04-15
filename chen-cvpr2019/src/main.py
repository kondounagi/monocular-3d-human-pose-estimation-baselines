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
from martinez_model import MartinezModel
from datamodule import CustomDataModule
from mpjpe import MPJPE, P_MPJPE


class PoseNet(pl.LightningModule):
    def __init__(
        self,
        gen_lr: float = 0.001,
        dis_lr: float = 0.001,
        gan_accuracy_cap: float = 0.9,
        heuristic_loss_weight: float = 0.5,  # XXX: source ?
        use_heuristic_loss: bool = False,
        use_sh_detection: bool = False,
        n_in: int = 34,
        n_unit: int = 1024,
        mode: str = "unsupervised",
        activate_func=F.leaky_relu,
        model_type: str = "martinez",
    ):
        super().__init__()
        self.save_hyperparameters()

        # set generator and parameter
        keys_to_remove = ("gen_lr", "dis_lr")

        gen_hparams = copy.deepcopy(self.hparams)
        gen_hparams["mode"] = "generator"
        # gen_hparams.pop(keys_to_remove)

        dis_hparams = copy.deepcopy(self.hparams)
        dis_hparams["mode"] = "discriminator"
        # dis_hparams.pop(keys_to_remove)

        if self.hparams.model_type == "martinez":
            gen_hparams["num_stage"] = 4
            dis_hparams["num_stage"] = 3
            self.gen = MartinezModel(**gen_hparams)
            self.dis = MartinezModel(**dis_hparams)
        elif self.hparams.moddel_type == "kudo":
            self.gen = KudoModel(**gen_hparams)
            self.dis = KudoModel(**dis_hparams)
        else:
            raise NotImplementedError

        self.val_mpjpe = MPJPE()
        self.val_p_mpjpe = P_MPJPE()
        self.test_mpjpe = MPJPE()
        self.test_p_mpjpe = P_MPJPE()

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # xy_real: joint_num * 2(xy axis)
        # xy_proj, xyz, scale = batch
        xy_proj, xyz, scale = batch
        batch_size = len(xy_proj)
        xy_real, xyz = xy_proj[:, 0], xyz[:, 0]
        z_real = xyz.narrow(-1, 2, 1)
        z_pred = self(xy_real)
        z_mse = F.mse_loss(z_pred, z_real)
        if self.hparams.mode == "supervised":
            return z_mse
        elif self.hparams.mode == "unsupervised":
            # Random rotation
            # TODO: [0, 2pi) の一様分布をautogradありで生成する方法のベストプラクティスを探す
            theta = torch.rand(batch_size, 1) * 2 * np.pi
            cos_theta = torch.cos(theta).to(self.device)
            sin_theta = torch.sin(theta).to(self.device)

            xy_real = xy_real.view(batch_size, 17, 2)
            x = xy_real.narrow(-1, 0, 1).squeeze()
            y = xy_real.narrow(-1, 1, 1).squeeze()
            # y = xy_real[:, 1::2]
            new_x = x * cos_theta + z_pred * sin_theta
            xyz_fake = torch.stack((new_x, y, z_pred), dim=2)
            xyz_real = xyz

            y_real = self.dis(xyz_real.view(batch_size, -1))
            y_fake = self.dis(xyz_fake.view(batch_size, -1))

            acc_dis_fake = pl.metrics.functional.accuracy(
                y_fake, torch.zeros_like(y_fake, dtype=torch.int).to(self.device)
            )
            acc_dis_real = pl.metrics.functional.accuracy(
                y_real, torch.ones_like(y_real, dtype=torch.int).to(self.device)
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

            if optimizer_idx == 0:
                if acc_dis >= (1 - self.hparams.gan_accuracy_cap):
                    return loss_gen
                else:
                    pass
            elif optimizer_idx == 1:
                if acc_dis < self.hparams.gan_accuracy_cap:
                    return loss_dis
                else:
                    pass
            else:
                return NotImplementedError

    def validation_step(self, batch, batch_idx):
        xy_proj, xyz, scale = batch
        xy_real, xyz = xy_proj[:, 0], xyz[:, 0]
        z_pred = self(xy_real)
        loss = F.mse_loss(z_pred, xyz.narrow(-1, 2, 1))
        self.log("val_loss_step", loss)

        batch_size = len(xyz)
        xyz_pred = torch.cat(
            (xy_real, z_pred),
            dim=-1,
        )
        # xyz = xyz.view(batch_size, 17, 3)
        # NOTE: datamodule内でのscaleを反映する必要はないか？
        self.log("val_mpjpe_step", self.val_mpjpe(xyz_pred, xyz))
        self.log("val_p_mpjpe_step", self.val_p_mpjpe(xyz_pred, xyz))

    def validation_epoch_end(self, val_step_outputs):
        self.log("val_mpjpe_epoch", self.val_mpjpe.compute())
        self.log("val_p_mpjpe_epoch", self.val_p_mpjpe.compute())

    def test_step(self, batch, batch_idx):
        xy_proj, xyz, scale = batch
        xy_real, xyz = xy_proj[:, 0], xyz[:, 0]
        z_pred = self(xy_real)
        loss = F.mse_loss(z_pred, xyz.narrow(-1, 2, 1))
        self.log("test_loss_step", loss)

        batch_size = len(xyz)
        xyz_pred = torch.cat(
            (xy_real, z_pred),
            dim=-1,
        )
        xyz = xyz.view(batch_size, 17, 3)
        self.log("test_mpjpe_step", self.test_mpjpe(xyz_pred, xyz))
        self.log("test_p_mpjpe_step", self.test_p_mpjpe(xyz_pred, xyz))

    def test_epoch_end(self, test_step_outputs):
        self.log("test_mpjpe_epoch", self.test_mpjpe.compute())
        self.log("test_p_mpjpe_epoch", self.test_p_mpjpe.compute())

    def configure_optimizers(self):
        # TODO: change lr between dis and gen
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.gen_lr)
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=self.hparams.dis_lr)
        return [opt_g, opt_d], []

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
    parser = CustomDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mpjpe_epoch",
        filename="best-model-{epoch:02d}-val-mpjpe-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    dm = CustomDataModule(
        use_sh_detection=args.use_sh_detection, batch_size=args.batch_size
    )

    model = PoseNet()

    mlf_logger = MLFlowLogger(
        experiment_name="chen-cvpr2019", tracking_uri="file:./ml-runs"
    )
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], max_epochs=100
    )
    # args, callbacks=[checkpoint_callback], logger=mlf_logger, auto_lr_find=True)
    # args, callbacks=[checkpoint_callback], auto_lr_find=True, max_epochs=3)
    trainer.tune(model, datamodule=dm)
    trainer.fit(model, dm)

    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == "__main__":
    cli_main()
