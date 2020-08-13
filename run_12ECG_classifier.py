#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
from ECGDataset import ECGDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_12ECG_classifier(data, header_data, loaded_model):
    data = [data]
    model = loaded_model
    model.eval()
    test_dataset = ECGDataset(data, label=[0], headers=[header_data], train=False)
    test_loader = DataLoader(test_dataset, batch_size=1)
    data, label, age, sex = iter(test_loader).next()
    data = data.to(device)
    age = age.to(device)
    sex = sex.to(device)

    current_score = model.forward(data, age, sex).cpu().detach().numpy()[0]
    current_label = current_score > 0.15
    if sum(current_label)==0:
        current_label[21]=1
    classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003',
                      '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007', '111975006',
                      '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000', '63593006',
                      '164934002', '59931005', '17338001'])
    return current_label, current_score, classes


def load_12ECG_model(input_directory):
    loaded_model = torch.load(input_directory + '/net1.pkl')
    return loaded_model
