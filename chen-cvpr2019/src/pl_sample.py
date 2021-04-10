from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule


class PoseNet(pl.LightningModule):

    def __init__(self, n_in=34, n_unit=1024, mode='supervised',
                 use_bn=False, activate_func=F.reaky_relu):
        super().__init__()
        self.save_hyperparameters()

        n_out = n_in // 2 if mode == 'generator' else 1
        print('Model: {}, N_OUT{}, N_UNIT{}'.format(node, n_out, n_unit))
        self.mode = mode
        self.use_bn = use_bn
        self.activate_func = activate_func
        self.l1 = nn.Linear(self.hparams.n_in, self.hparams.n_unit)
        self.l2 = nn.Linear(self.hparams.n_unit, self.hparams.n_unit)
        self.l3 = nn.Linear(self.hparams.n_unit, self.hparams.n_unit)
        self.l4 = nn.Linear(self.hparams.n_unit, self.hparams.n_out)
        if self.hparams.use_bn:
            self.bn1 = nn.BatchNorm1d(self.hparams.n_unit)
            self.bn2 = nn.BatchNorm1d(self.hparams.n_unit)
            self.bn3 = nn.BatchNorm1d(self.hparams.n_unit)

        self.train_metrics = nn.ModuleDict({''})
        self.val_metrics = nn.ModuleDict({''})
        self.test_metrics = nn.ModuleDict({''})


    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.use_bn:
            h1 = self.activate_func(self.bn1(self.l1(x)))
            h2 = self.activate_func(self.bn2(self.l2(h1)))
            h3 = self.activate_func(self.bn3(self.l3(h2)) + h1)
        else:
            h1 = self.activate_func(self.l1(x))
            h2 = self.activate_func(self.l2(h1))
            h3 = self.activate_func(self.l3(h2) + h1)
        return self.l4(h3)

    def training_step(self, batch, batch_idx):
        # xy_real: joint_num * 2(xy axis)
        xy_real, z_real = batch
        batch_size = len(xy_real)
        z_pred = self(xy_real)
        z_mse = F.mse_loss(z_pred, z_real)
        if self.mode == 'supervised':
            loss = z_mse
        elif self.mode == 'unsupervised':
            # Random rotation
            # TODO: [0, 2pi) の一様分布をautogradありで生成する方法のベストプラクティスを探す
            theta = torch.rand(1) * 2 * np.pi
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            # 2D Projection
            x = xy_real[:, 0::2]
            y = xy_real[:, 1::2]
            new_x = x * cos_theta + z_pred * sin_theta
            xy_fake = torch.concat((new_x, y), dim=2).view(batch_size, -1)

            acc_dis_fake = F.binar
            acc_dis_real

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

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
    def calculate_heuristic_loss(xy_real, z_pred):
        return torch.mean(F.relu(-self.calculate_rotation(xy_real, z_pred)))


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    dm = MNISTDataModule.from_argparse_args(args)

    model = LitClassifier(args.hidden_dim, args.learning_rate)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)

    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()

