import torch
import torch.nn as nn
import torch_geometric

import math
import pandas as pd
import numpy as np

from copy import deepcopy
from itertools import chain

import rdkit
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
import networkx as nx

from tqdm import tqdm
import datetime
import random

from .embedding_functions_qm9 import embedConformerWithAllPaths

class MaskedGraphDataset(torch_geometric.data.Dataset):
    def __init__(self, df, regression = '', stereoMask = True, mask_coordinates = False):
        super(MaskedGraphDataset, self).__init__()
        self.df = df
        self.stereoMask = stereoMask
        self.mask_coordinates = mask_coordinates
        self.regression = regression
        
    def get_all_paths(self, G, N = 3):
        # adapted from: https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph
        def findPaths(G,u,n):
            if n==0:
                return [[u]]
            paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
            return paths
    
        allpaths = []
        for node in G:
            allpaths.extend(findPaths(G,node,N))
        return allpaths
    
    def process_mol(self, mol):
        # get internal coordinates for conformer, using all possible (forward) paths of length 2,3,4
        # Reverse paths (i.e., (1,2) and (2,1) or (1,2,3,4) and (4,3,2,1)) are not included when repeats == False
        # Note that we encode the reverse paths manually in alpha_encoder.py
        
        atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_index, bond_angles, bond_angle_index, dihedral_angles, dihedral_angle_index = embedConformerWithAllPaths(mol, repeats = False)
        
        bond_angles = bond_angles % (2*np.pi)
        dihedral_angles = dihedral_angles % (2*np.pi)
        
        data = torch_geometric.data.Data(x = torch.as_tensor(node_features), edge_index = torch.as_tensor(edge_index, dtype=torch.long), edge_attr = torch.as_tensor(edge_features))
        data.bond_distances = torch.as_tensor(bond_distances)
        data.bond_distance_index = torch.as_tensor(bond_distance_index, dtype=torch.long).T
        data.bond_angles = torch.as_tensor(bond_angles)
        data.bond_angle_index = torch.as_tensor(bond_angle_index, dtype=torch.long).T
        data.dihedral_angles = torch.as_tensor(dihedral_angles)
        data.dihedral_angle_index = torch.as_tensor(dihedral_angle_index, dtype=torch.long).T
        
        return data
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, key):
        mol = deepcopy(self.df.iloc[key].rdkit_mol)
        
        data = self.process_mol(mol)
        
        if self.regression != '':
            #self.regression is the variable name of the supervised target in self.df
            y = torch.tensor(deepcopy(self.df.iloc[key][self.regression])) 

        if self.stereoMask:
            data.x[:, -9:] = 0.0
            data.edge_attr[:, -7:] = 0.0

        if self.mask_coordinates:
            data.bond_distances[:] = 0.0
            data.bond_angles[:] = 0.0
            data.dihedral_angles[:] = 0.0

        return (data, y) if self.regression != '' else data

