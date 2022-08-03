import torch
import torch.nn as nn
import torch_geometric
import datetime
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .train_functions import regression_loop_alpha

def train_regression_model(model, train_loader, val_loader, N_epochs, optimizers, device, batch_size, scheduler = None, auxillary_torsion_loss = 0.02, save = True, PATH = ''):
    
    train_epoch_losses = []
    train_epoch_aux_losses = []
    train_epoch_mse = []
    #train_epoch_mae = []
    
    val_epoch_losses = []
    val_epoch_aux_losses = []
    val_epoch_mse = []
    val_epoch_mae = []
    
    best_val_mse = np.inf
    best_epoch = 0
    
    for epoch in tqdm(range(1, N_epochs+1)):
    
        train_losses, train_aux_losses, train_batch_sizes, train_batch_mse, train_batch_mae = regression_loop_alpha(model, train_loader, optimizers, device, epoch, batch_size, training = True, auxillary_torsion_loss = auxillary_torsion_loss)

        epoch_loss = torch.mean(torch.tensor(train_losses))
        epoch_aux_loss = torch.mean(torch.tensor(train_aux_losses))
        train_mse = torch.mean(torch.tensor(train_batch_mse))
        #train_mae = torch.mean(torch.tensor(train_batch_mae))

        train_epoch_losses.append(epoch_loss)
        train_epoch_aux_losses.append(epoch_aux_loss)
        train_epoch_mse.append(train_mse)
        #train_epoch_mae.append(train_mae)

        with torch.no_grad():
            val_losses, val_aux_losses, val_batch_sizes, val_batch_mse, val_batch_mae = regression_loop_alpha(model, val_loader, optimizers, device, epoch, batch_size, training = False, auxillary_torsion_loss = auxillary_torsion_loss)
            
            #print('RMSE:', val_batch_rmse)
            #print('MAE:', val_batch_mae)
 
            val_epoch_loss = torch.mean(torch.tensor(val_losses))
            val_epoch_aux_loss = torch.mean(torch.tensor(val_aux_losses))
            val_mse = torch.mean(torch.tensor(val_batch_mse))
            val_mae = torch.mean(torch.tensor(val_batch_mae))
            
            if scheduler is not None:
                scheduler.step(val_epoch_loss)
                print(f"lr: {optimizers[0].param_groups[0]['lr']}")
                if scheduler.optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
                    break

            val_epoch_losses.append(val_epoch_loss)
            val_epoch_aux_losses.append(val_epoch_aux_loss)
            val_epoch_mse.append(val_mse)
            val_epoch_mae.append(val_mae)
        
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if save == True:
                    torch.save(model.state_dict(), PATH + 'best_model.pt')
                    print('\n    saving best model:' + str(epoch))
                    print('    Best Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss, 'Validation MAE', val_mae, 'Validation Aux. Loss', val_epoch_aux_loss)

            if epoch % 1 == 0:
                print('Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss)
                print('    Epoch:', epoch, 'Train Loss:', epoch_loss, 'Validation Loss:', val_epoch_loss, 'Validation MAE', val_mae, 'Validation Aux. Loss', val_epoch_aux_loss)
                if (save == True) and (epoch % 5 == 0):
                    torch.save(model.state_dict(), PATH + 'checkpoint_models/' + 'checkpoint_model_' + str(epoch) + '.pt')
                    torch.save(train_epoch_losses, PATH + 'train_epoch_losses.pt')
                    torch.save(val_epoch_losses, PATH + 'val_epoch_losses.pt')
                    torch.save(train_epoch_aux_losses, PATH + 'train_epoch_aux_losses.pt')
                    torch.save(val_epoch_aux_losses, PATH + 'val_epoch_aux_losses.pt')
    
    return best_state_dict


