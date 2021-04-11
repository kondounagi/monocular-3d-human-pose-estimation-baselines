import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class MPIIDataModule(LightningDataModule):
    def __init__(self, use_sh_detection: bool):
        super().__init__()
        self.use_sh_detection = use_sh_detection

    # def prepare_data(self):
    #     # download, split, etc...
    #     # only called on 1 GPU/TPU in distributed
    # def setup(self):
    #     # make assignments here (val/train/test split)
    #     # called on every process in DDP
    def train_dataloader(self):
        train_split = MPII(train=True, use_sh_detection=self.use_sh_detection)
        return DataLoader(train_split)

    def val_dataloader(self):
        val_split = MPII(train=False, use_sh_detection=self.use_sh_detection)
        return DataLoader(val_split)

    # def test_dataloader(self):
    # def teardown(self):
    # clean up after fit or test
    # called on every process in DDP


class MPII(Dataset):
    def __init__(self, train=True, use_sh_detection=False):
        if use_sh_detection:
            raise NotImplementedError
        else:
            self.poses = np.load("data/mpii_poses.npy")

        np.random.seed(100)
        perm = np.random.permutation(len(self.poses))
        if train:
            self.poses = self.poses[perm[: int(len(self.poses) * 0.9)]]
        else:
            self.poses = self.poses[perm[int(len(self.poses) * 0.9) :]]

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        mpii_poses = self.poses[idx : idx + 1]
        # hip(0)と各関節点の距離の平均値が1になるようにスケール
        xs = mpii_poses.T[0::2] - mpii_poses.T[0]
        ys = mpii_poses.T[1::2] - mpii_poses.T[1]
        # hip(0)が原点になるようにシフト
        mpii_poses = mpii_poses.T / np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
        mpii_poses[0::2] -= mpii_poses[0]
        mpii_poses[1::2] -= mpii_poses[1]
        mpii_poses = mpii_poses.T.astype(np.float32)[None]

        dummy_X = np.zeros((1, 1, 17 * 3), dtype=np.float32)
        dummy_X[0, 0, 0::3] = mpii_poses[0, 0, 0::2]
        dummy_X[0, 0, 1::3] = mpii_poses[0, 0, 1::2]
        dummy_scale = np.array([1], dtype=np.float32)

        return mpii_poses, dummy_X, dummy_scale
