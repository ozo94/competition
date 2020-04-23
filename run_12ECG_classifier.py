#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
import lightgbm as lgb
import pandas as pd

def run_12ECG_classifier(data,header_data,classes,model_lst):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    threshold_=[0.177548, 0.104988, 0.034317, 0.133488, 0.270031, 0.089574, 0.101789, 0.126363, 0.031991]
    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)
    for i in range(num_classes):
        model=model_lst[i]
        threshold=threshold_[i]
        score = model.predict(feats_reshape)
        if score > threshold:
            label = 1
        else:
            label=0

        current_label[i] = label
        current_score[i] = score

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    labels_=['AF', 'I-AVB', 'LBBB', 'Normal', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    loaded_model=[]
    for label in labels_:
        
        filename='./lgb_model/{}_model.txt'.format(label)
#         print(filename)
        loaded_model.append(lgb.Booster(model_file=filename))
        
    return loaded_model
