from argparse import ArgumentParser
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule


class PoseNet(pl.LightningModule):

    def __init__(self, n_in=34, n_unit=1024, mode='generator',
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
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitClassifier")
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = MNISTDataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    model = LitClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()

