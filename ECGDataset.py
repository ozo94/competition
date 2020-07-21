# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
import numpy as np
import torch


class ECGDataset(Dataset):
    def __init__(self, data, label):
        super(ECGDataset, self).__init__()
        self.data = data
        self.label = label
        self.data_lens = 5000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        data_lens = data.shape[1]
        if data_lens < self.data_lens:
            data = np.pad(data, ((0, 0), (self.data_lens - data_lens, 0)), constant_values=0)
        elif data_lens > self.data_lens:
            data = data[:, :self.data_lens]

        label = self.label[item]

        data = torch.tensor(np.array(data), dtype=torch.float)
        label = torch.tensor(np.array(label), dtype=torch.float)

        return data, label
