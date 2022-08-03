import torch
import torch.nn as nn
import numpy as np
import math
import torch_scatter

def MSE(y, y_hat):
    MSE = torch.mean(torch.square(y - y_hat))
    return MSE

