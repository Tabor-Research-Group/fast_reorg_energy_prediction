import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import datetime
import scipy
import gzip
import math
import rdkit
import rdkit.Chem
from rdkit.Chem import TorsionFingerprints
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import random
import pickle

import os
import sys
import json
from model.params_interpreter import string_to_object 

from model.alpha_encoder import Encoder

from model.train_functions import regression_loop_alpha

from model.train_functions import evaluate_regression_loop_alpha

from model.train_models import train_regression_model

from model.datasets_samplers import MaskedGraphDataset

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

args = sys.argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('reading data...')

# READ HYPERPARAMETERS
with open(str(args[1])) as f: # args[1] should contain path to params.json file
    params = json.load(f)

# set random seed for getting the testing set
test_seed = 37
random.seed(test_seed)

# read the smiles list to get same split with SchNet
smiles_list = pickle.load(open(params['smileslist'],'rb'))
train_size = int(len(smiles_list) * 0.8)
val_size = int(len(smiles_list) * 0.1)
random.shuffle(smiles_list)
test_smiles = smiles_list[train_size + val_size:]

# LOAD GEOMETRIES (DFT, CREST, RDKit) AND TARGETS (IP, EA, or REORG. ENERGY)
full_dataframe = pd.read_pickle(params['datafile'])
test_dataframe = full_dataframe[full_dataframe.smiles.apply(lambda x: x in test_smiles)] 
#test_dataframe = test_dataframe.groupby('smiles').sample(1, random_state = seed) # this line is for direct reorg. energy prediction using rdkit_qm9_reorg_5repeat.pkl

# CREATE DIRECTORY FOR SAVING/CHECKPOINTING
save = params['save']

PATH = args[2] # should contain path to subfolder where files will be saved
if PATH[-1] != '/':
    PATH = PATH + '/'

if not os.path.exists(PATH) and save == True:
    os.makedirs(PATH)

#CREATE MODEL
seed = params['random_seed']
random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)

print('creating model...')
layers_dict = deepcopy(params['layers_dict'])

activation_dict = deepcopy(params['activation_dict'])
for key, value in params['activation_dict'].items(): 
    activation_dict[key] = string_to_object[value] # convert strings to actual python objects/functions using pre-defined mapping

num_node_features = 21
num_edge_features = 7

model = Encoder(
    F_z_list = params['F_z_list'], # dimension of latent space
    F_H = params['F_H'], # dimension of final node embeddings, after EConv and GAT layers
    F_H_embed = num_node_features, # dimension of initial node feature vector, currently 21
    F_E_embed = num_edge_features, # dimension of initial edge feature vector, currently 7
    F_H_EConv = params['F_H_EConv'], # dimension of node embedding after EConv layer
    layers_dict = layers_dict,
    activation_dict = activation_dict,
    GAT_N_heads = params['GAT_N_heads'],
    chiral_message_passing = params['chiral_message_passing'],
    c_coefficient_normalization = params['c_coefficient_normalization'], # None, or one of ['softmax']
    encoder_reduction = params['encoder_reduction'], #mean or sum
    output_concatenation_mode = params['output_concatenation_mode'], 
    EConv_bias = params['EConv_bias'], 
    GAT_bias = params['GAT_bias'], 
    encoder_biases = params['encoder_biases'], 
    dropout = params['dropout'], # applied to hidden layers (not input/output layer) of Encoder MLPs, hidden layers (not input/output layer) of EConv MLP, and all GAT layers (using their dropout parameter)
    )

# LOADING PRETRAINED MODEL
if params['pretrained'] != "":
    print('loading pretrained weights...')
    model.load_state_dict(torch.load(params['pretrained'], map_location=next(model.parameters()).device), strict=False)

model.to(device)

num_workers = params['num_workers']

# BUILDING DATA LOADERS
test_dataset = MaskedGraphDataset(test_dataframe, 
                                    regression = 'vertical_IP', # EA, REORG. ENERGY
                                    stereoMask = False,
                                    mask_coordinates = False, 
                                    )

test_loader = torch_geometric.loader.DataLoader(test_dataset, num_workers = num_workers, batch_size = 1000, shuffle = False)

# BEGIN EVALUATION
targets, outputs = evaluate_regression_loop_alpha(model, test_loader, device, batch_size = 1000, dataset_size = len(test_dataset))

print('MAE: ', mean_absolute_error(targets, outputs))
print('RMSE: ', mean_squared_error(targets, outputs, squared=False))
print('MAE: ', r2_score(targets, outputs))

