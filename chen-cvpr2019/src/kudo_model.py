import torch
from torch import nn
from torch.nn import functional as F


class KudoModel(nn.Module):
    def __init__(
        self,
        n_in: int = 34,
        n_unit: int = 1024,
        mode: str = "supervised",
        use_bn: bool = True,
        activate_func=F.leaky_relu,
        **kwargs
    ):
        super().__init__()
        self.n_in = n_in
        self.n_unit = n_unit
        self.mode = mode
        self.use_bn = use_bn
        self.activate_func = activate_func
        self.n_out = n_in // 2 if mode == "generator" else 1

        self.l1 = nn.Linear(self.n_in, self.n_unit)
        self.l2 = nn.Linear(self.n_unit, self.n_unit)
        self.l3 = nn.Linear(self.n_unit, self.n_unit)
        self.l4 = nn.Linear(self.n_unit, self.n_out)

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(self.n_unit)
            self.bn2 = nn.BatchNorm1d(self.n_unit)
            self.bn3 = nn.BatchNorm1d(self.n_unit)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.use_bn:
            h1 = self.activate_func(self.bn1(self.l1(x)))
            h2 = self.activate_func(self.bn2(self.l2(h1)))
            h3 = self.activate_func(self.bn3(self.l3(h2)) + h1)
            h4 = self.l4(h3)
            if self.mode == "discriminator":
                h4 = torch.sigmoid(h4)
        else:
            h1 = self.activate_func(self.l1(x))
            h2 = self.activate_func(self.l2(h1))
            h3 = self.activate_func(self.l3(h2) + h1)
            h4 = self.l4(h3)
            if self.mode == "discriminator":
                h4 = torch.sigmoid(h4)
        return h4