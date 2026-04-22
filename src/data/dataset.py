import numpy as np
import torch
from torch.utils.data import Dataset

from src.config.config import Config


class HyperCDDataset(Dataset):
    def __init__(self, t1, t2, label, text, index, transform=None):
        """

        :param t1: c h w
        :param t2: c h w
        :param label: h w
        :param text: (cls_num, token_len)
        :param index:
        :param transform:
        """
        super(HyperCDDataset, self).__init__()
        data_param = Config.args.data
        self.t1 = t1
        self.t2 = t2
        self.label = label
        self.text = text
        self.index = index
        self.transform = transform
        self.patch_size = data_param.patch_size

    def __getitem__(self, index):
        pix_idx_1d = self.index[index]
        pix_idx_2d = np.unravel_index(pix_idx_1d, self.label.shape)
        t1_patch = self.t1[:, pix_idx_2d[0]:pix_idx_2d[0] + self.patch_size, pix_idx_2d[1]:pix_idx_2d[1] + self.patch_size]
        t2_patch = self.t2[:, pix_idx_2d[0]:pix_idx_2d[0] + self.patch_size, pix_idx_2d[1]:pix_idx_2d[1] + self.patch_size]
        label = self.label[pix_idx_2d]
        if self.transform:
            # 保证双时相经过相同变换
            img = torch.stack([t1_patch, t2_patch])
            img = self.transform(img)
            t1_patch = img[0]
            t2_patch = img[1]
        return t1_patch, t2_patch, label, self.text

    def __len__(self):
        return len(self.index)
