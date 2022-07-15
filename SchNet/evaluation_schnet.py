#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import schnetpack as spk
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load input geometries (DFT or CREST or RDKit) along with their target properties (IP or EA)
db = spk.AtomsData('data/SchNet/rdkit_qm9_vertical_ip.db')
# load split.npz from dumped from training stage to get the same train/val/test splits
train, val, test = spk.train_test_split(
        data=db,
        split_file=os.path.join("./", "split.npz"),
    )

test_loader = spk.AtomsLoader(test, batch_size=1000)

# BEGIN EVALUATION
pred_list = list()
target_list = list()
for count, batch in enumerate(test_loader):
    batch = {k: v.to(device) for k, v in batch.items()}

    pred = best_model(batch)
    
    target_list += torch.reshape(batch['IP'], (-1,)).tolist() # EA
    pred_list += torch.reshape(pred['IP'], (-1,)).tolist() # EA

print('MAE: ', mean_absolute_error(target_list, pred_list))
print('RMSE: ', mean_squared_error(target_list, pred_list, squared=False))
print('MAE: ', r2_score(target_list, pred_list))

