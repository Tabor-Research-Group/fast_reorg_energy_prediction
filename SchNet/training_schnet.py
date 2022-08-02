#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import schnetpack as spk
from torch.optim import Adam
import schnetpack.train as trn
import numpy as np
import pandas as pd
import torch
import random
import pickle
import copy

# set random seed for creating testing test
test_seed = 37
random.seed(test_seed)

# read the smiles list to make sure same split with ChiRo
origin_smiles_list = pickle.load(open('data/smiles_list.pkl','rb')) # change this to crest_ip_smiles_list.pkl, crest_ea_smiles_list.pkl or rdkit_smiles_list.pkl for CREST or RDKit geometries
smiles_list = pickle.load(open('data/smiles_list.pkl','rb'))  
train_size = int(len(smiles_list) * 0.8)
val_size = int(len(smiles_list) * 0.1)
random.shuffle(smiles_list)
train_val_smiles = smiles_list[:train_size + val_size]
test_smiles = smiles_list[train_size + val_size:]

# set random seed for training and training/validation splits
seed = 1
random.seed(seed)

random.shuffle(train_val_smiles)
train_smiles = train_val_smiles[:train_size]
val_smiles = train_val_smiles[train_size:]

train_idx = [origin_smiles_list.index(smiles) for smiles in train_smiles]
val_idx = [origin_smiles_list.index(smiles) for smiles in val_smiles]
test_idx = [origin_smiles_list.index(smiles) for smiles in test_smiles]
np.savez(open('split.npz','wb'), train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

# LOAD GEOMETRIES (DFT or CREST) AND TARGETS (IP or EA)
db = spk.AtomsData('data/SchNet/dft_qm9_vertical_ip.db')
device = "cuda"

train, val, test = spk.train_test_split(
        data=db,
        split_file=os.path.join("./", "split.npz"),
    )

train_loader = spk.AtomsLoader(train, batch_size=128, num_workers=8, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=128, num_workers=8, shuffle=True)

means, stddevs = train_loader.get_statistics(
    'IP', divide_by_atoms=False
) #EA

# CREATE MODEL
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

schnet = spk.representation.SchNet(
    n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
    cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
)

output_IP = spk.atomistic.Atomwise(n_in=30, property='IP', mean=means['IP'], stddev=stddevs['IP'], aggregation_mode='avg') # EA 

model = spk.AtomisticModel(representation=schnet, output_modules=output_IP)

# DEFINE LOSS FUNCTION, OPTIMIZER, AND SCHEDULER
loss = trn.build_mse_loss(['IP']) # EA

optimizer = Adam(model.parameters(), lr=1e-2)

metrics = [spk.metrics.MeanAbsoluteError('IP')] # EA
hooks = [
    trn.CSVHook(log_path='./', metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

# BEGIN TRAINING
trainer = trn.Trainer(
    model_path='./',
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

trainer.train(device=device)

