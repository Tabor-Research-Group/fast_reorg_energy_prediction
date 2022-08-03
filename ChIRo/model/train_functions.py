import torch
import torch.nn as nn
import torch_geometric
import datetime
import numpy as np
from tqdm import tqdm
import math
from collections import OrderedDict

from .optimization_functions import MSE

from itertools import chain

import random

def get_local_structure_map(psi_indices):
    LS_dict = OrderedDict()
    LS_map = torch.zeros(psi_indices.shape[1], dtype = torch.long)
    v = 0
    for i, indices in enumerate(psi_indices.T):
        tupl = (int(indices[1]), int(indices[2]))
        if tupl not in LS_dict:
            LS_dict[tupl] = v
            v += 1
        LS_map[i] = LS_dict[tupl]

    alpha_indices = torch.zeros((2, len(LS_dict)), dtype = torch.long)
    for i, tupl in enumerate(LS_dict):
        alpha_indices[:,i] = torch.LongTensor(tupl)

    return LS_map, alpha_indices

def regression_loop_alpha(model, loader, optimizers, device, epoch, batch_size, training = True, auxillary_torsion_loss = 0.02):
    if training:
        model.train()
    else:
        model.eval()

    batch_losses = []
    batch_aux_losses = []
    batch_sizes = []
    batch_mse = []
    batch_mae = []
    
    for batch in loader:
        batch_data, y = batch
        y = y.type(torch.float32)
        
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)
        y = y.to(device)

        if training:
            for opt in optimizers:
                opt.zero_grad()
        
        output, latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)
        
        aux_loss = torch.mean(torch.abs(1.0 - phase_shift_norm.squeeze()))
        loss = MSE(y.squeeze(), output.squeeze())
        backprop_loss = loss + aux_loss*auxillary_torsion_loss
        
        mse = loss.detach()
        mae  = torch.mean(torch.abs(y.squeeze().detach() - output.squeeze().detach()))
        #acc = 1.0 - (torch.sum(torch.abs(y.squeeze().detach() - torch.round(torch.sigmoid(output.squeeze().detach())))) / y.shape[0])
        
        if training:
            backprop_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        
            for opt in optimizers:
                opt.step()
        
        batch_sizes.append(y.shape[0])
        batch_losses.append(loss.item())
        batch_aux_losses.append(aux_loss.item())
        batch_mse.append(mse.item())
        batch_mae.append(mae.item())
         
        
    return batch_losses, batch_aux_losses, batch_sizes, batch_mse, batch_mae


def evaluate_regression_loop_alpha(model, loader, device, batch_size, dataset_size):
    model.eval()
    
    all_targets = torch.zeros(dataset_size).to(device)
    all_outputs = torch.zeros(dataset_size).to(device)
    
    start = 0
    for batch in loader:
        batch_data, y = batch
        y = y.type(torch.float32)
        
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        batch_data = batch_data.to(device)
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)
        y = y.to(device) 

        with torch.no_grad():
            output, latent_vector, phase_shift_norm, z_alpha, mol_embedding, c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = model(batch_data, LS_map, alpha_indices)
            
            all_targets[start:start + y.squeeze().shape[0]] = y.squeeze()
            all_outputs[start:start + y.squeeze().shape[0]] = output.squeeze()
            start += y.squeeze().shape[0]
       
    return all_targets.detach().cpu().numpy(), all_outputs.detach().cpu().numpy()


