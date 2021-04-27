# ref. https://openaccess.thecvf.com/content_ICCV_2017/papers/Martinez_A_Simple_yet_ICCV_2017_paper.pdf
import torch
from torch import nn
from torch.nn import functional as F


class MartinezModel(nn.Module):
    def __init__(
        self,
        n_unit: int = 1024,
        mode: str = "generator",
        num_stage: int = 2,
        p_dropout: float = 0.5,
        **kwargs
    ):
        super(MartinezModel, self).__init__()

        self.n_unit = n_unit
        self.num_stage = num_stage
        self.p_dropout = p_dropout
        self.mode = mode
        joint_num = 17

        if self.mode == "generator":
            self.input_size = joint_num * 2
            self.output_size = joint_num
        elif self.mode == "discriminator":
            # discriminator mode
            self.input_size = joint_num * 3
            self.output_size = 1
        else:
            raise NotImplementedError

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.n_unit)
        self.batch_norm1 = nn.BatchNorm1d(self.n_unit)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(Linear(self.n_unit, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.n_unit, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        # TODO: main側のスクリプトを修正してsqueezeを除く
        y = self.batch_norm1(y.squeeze())
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        if self.mode == "discriminator":
            y = torch.sigmoid(y)

        return y


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, n_unit, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = n_unit
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out
