# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from scipy import signal
import copy


class ECGDataset(Dataset):
    def __init__(self, data, label, headers, train=False):
        super(ECGDataset, self).__init__()
        self.data = data
        self.label = label
        self.headers = headers
        self.data_lens = 5000
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        # age、sex
        header = self.headers[item]
        line0 = header[0].split()
        if int(line0[2]) != 500:
            tmp_data = []
            nums = int(int(line0[3]) / int(line0[2]) * 500)
            for i in range(data.shape[0]):
                tmp_data.append(signal.resample(data[i], nums))
            data = copy.deepcopy(np.array(tmp_data))
        for lines in header:
            if lines.startswith('#Age'):
                try:
                    age = int(lines.strip().split(': ')[1])
                except:
                    age = 0
            if lines.startswith('#Sex'):
                sex = lines.strip().split(': ')[1]
                if sex == 'Female':
                    sex = 1
                elif sex == 'Male':
                    sex = 2
                else:
                    sex = 0

        data_lens = data.shape[1]
        if data_lens < self.data_lens:
            data = np.pad(data, ((0, 0), (self.data_lens - data_lens, 0)), mode='constant', constant_values=0)
        elif data_lens > self.data_lens:
            data = data[:, :self.data_lens]
        data = data_transform(data, train=self.train)
        label = self.label[item]

        data = torch.tensor(np.array(data), dtype=torch.float)
        label = torch.tensor(np.array(label), dtype=torch.float)
        age = torch.tensor(np.array(age), dtype=torch.float)
        sex = torch.tensor(np.array(sex), dtype=torch.float)

        return data, label, age, sex


def scaling(sig, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma,
                                     size=(1, sig.shape[1]))
    myNoise = np.matmul(np.ones((sig.shape[0], 1)), scalingFactor)
    return sig * myNoise


def shift(sig, interval=50):
    for col in range(sig.shape[0]):
        offset = np.random.choice(range(-interval, interval))
        sig[col, :] += offset
    return sig


def data_transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)

        if np.random.randn() > 0.5:
            sig = shift(sig)

    return sig
