#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
import pandas as pd
from ECGResNet import ResNet34
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ECGDataset import ECGDataset
from lr_scheduler import LRScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_12ECG_classifier(input_directory, output_directory):
    # Load data.
    print('Loading data...')
    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if not f.lower().startswith('.') and f.lower().endswith('hea') and os.path.isfile(g):
            header_files.append(g)

    classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003',
                      '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007', '111975006',
                      '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000', '63593006',
                      '164934002', '59931005', '17338001'])
    num_classes = len(classes)
    num_files = len(header_files)
    recordings = list()
    headers = list()

    for i in range(num_files):
        recording, header = load_challenge_data(header_files[i])
        recordings.append(recording)
        headers.append(header)

    # Train model.
    print('Training model...')

    labels = list()

    for i in range(num_files):
        header = headers[i]

        for l in header:
            if l.startswith('#Dx:'):
                labels_act = np.zeros(num_classes+1)
                arrs = l.strip().split(' ')
                for arr in arrs[1].split(','):
                    if arr.rstrip() in classes:
                        class_index = classes.index(arr.rstrip())  # Only use first positive index
                        labels_act[class_index] = 1
                labels_act[-1] = len(arrs[1].split(','))
        labels.append(labels_act)

    labels = pd.DataFrame(labels, columns=classes+['label_sum'],dtype='int')
    labels['426783006'] = labels.apply(lambda df:(1 if ((df['426783006']==1)&(df['label_sum']==1)) else 0),axis=1)
    labels['713427006'] = labels['713427006'] | labels['59118001']
    labels['59118001'] = labels['713427006'] | labels['59118001']
    labels['284470004'] = labels['284470004'] | labels['63593006']
    labels['63593006'] = labels['284470004'] | labels['63593006']
    labels['427172004'] = labels['427172004'] | labels['17338001']
    labels['17338001'] = labels['427172004'] | labels['17338001']
    labels = np.array(labels[classes])
    # Train the classifier
    model = ResNet34(num_classes=27).to(device)
    train_dataset = ECGDataset(recordings, labels, headers, train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    niters = len(train_loader)
    lr_scheduler = LRScheduler(optimizer, niters, Config)
    net1 = train(train_loader, model, optimizer, lr_scheduler, 16)

    # Save model.
    print('Saving model...')

    torch.save(net1, output_directory + '/net1.pkl')


def train(train_loader, model, optimizer, lr_scheduler, n_epoch):
    for epoch in range(n_epoch):
        model.train()
        for i, (data, label, age, sex) in enumerate(train_loader):
            lr_scheduler.update(i, epoch)
            data = data.to(device)
            age = age.to(device)
            sex = sex.to(device)
            label = label.to(device)
            outputs = model.forward(data, age, sex)
            loss = F.binary_cross_entropy(outputs, target=label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


class Config:
    epochs = 16

    lr_mode = 'cosine'
    base_lr = 0.00075
    warmup_epochs = 2
    warmup_lr = 0.0
    targetlr = 0.0
    weight_decay = 0.00001


# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header
