# 
# Federated person re-identification for decentralised model learning
# Guile Wu and Shaogang Gong
# 2020
#

from __future__ import print_function, division
import torch
import numpy as np
import time
import copy
import sys
import os

from utils.logging import Logger
from lib.weightAgg import *
from lib.localUpdate import LocalUpdateLM
from torch.utils.tensorboard import SummaryWriter
from utils.meters import AverageMeter




def FedReID_train(model, w_glob, opt, local_datasets, dict_users, dataloaders_val,dataloader_viper):
    # Model save directory
   
    writer = SummaryWriter('runs/{}'.format(opt.name) + time.strftime(".%m_%d_%H:%M:%S"))

    name = opt.name
    dir_name = os.path.join(opt.logs_dir, name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    model_pth = 'model_{}'.format(opt.name)+ time.strftime(".%m_%d_%H:%M:%S") + '.pth' # saved model
    model_saved = os.path.join(dir_name, model_pth) # model directory
   # sys.stdout = Logger(os.path.join(opt.logs_dir, 'log' + time.strftime(".%m_%d_%H:%M:%S") + '.txt')) # training log

    since = time.time() # training start time
    num_epochs = opt.global_ep # global communication epochs
    best_val_loss = None # validation for stablising model performance
    m = max(int(opt.frac * opt.nusers), 1) # selected client number in each global communication epoch (1 =< C*N <= N)

    w_all, w_tmp = [], [] # all local model parameters and local temporal model parameters
    for i in range(opt.nusers+1):
        w_all.append(w_glob) # initial


    loss_meter = AverageMeter('Total loss for selected clients', ':6.3f')

    # Training
    for epoch in range(num_epochs):
        print('Global Training Epoch {}/{}'.format(epoch+1, num_epochs))

        loss_meter.reset()
        w_locals= []# aggreated local model parameters and local loss

        idxs_users_selected = np.random.choice(range(opt.nusers), m, replace=False) # randomly selected clients

        # local client model updating
        for idx in idxs_users_selected:
            print('=====training global round {} for client {} =============='.format(epoch,idx))
            # local client model initialisation and local dataset partition
            # idxs: each local client only contains one user here
            local = LocalUpdateLM(args=opt, dataset=local_datasets[idx], idxs=dict_users[idx][0], nround=epoch, user=idx)

            # local client weight update
            w_tmp.append(w_glob) # server model parameters
            w_tmp.append(w_all[idx]) # local model parameters
            model.load_state_dict(weights_aggregate(w_tmp, opt.dp, opt.alpha_mu, idx_client=idx, is_local=True)) # local model parameter update

            # central model
            model_sv = copy.deepcopy(model)
            model_sv.load_state_dict(w_glob)

            # local client model training, return model parameters and training loss
            out_dict = local.update_weights(model=copy.deepcopy(model), cur_epoch=epoch,
                                            idx_client=idx, model_sv=copy.deepcopy(model_sv))

            # store updated local client parameters
            w_locals.append(copy.deepcopy(out_dict['params']))
            loss_meter.update(out_dict['loss_meter'])
            #shitong

            # store all local client parameters (some clients are not updated in the randomly selection)
            w_all[idx] = copy.deepcopy(out_dict['params'])
            w_tmp = [] # clear local temporal model parameters
            writer.add_scalar('baseline/client {} total loss'.format(idx),
                              out_dict['loss_meter'],
                              epoch)
            writer.add_scalar('baseline/client {} KL loss'.format(idx),
                              out_dict['KL_loss'],
                              epoch)
            writer.add_scalar('baseline/client {} local_loss'.format(idx),
                              out_dict['client_loss'],
                              epoch)
            writer.add_scalar('baseline/client {} server_loss'.format(idx),
                              out_dict['server_loss'],
                              epoch)
            writer.add_scalar('baseline/client {} accuracy'.format(idx),
                              out_dict['acc'],
                              epoch)

        # central server model updating 
        if opt.agg == 'avg': # current version  only supports modified federated average strategy
            w_glob = weights_aggregate(w_locals, opt.dp, opt.alpha_mu, idx_client=idxs_users_selected, is_local=False)#  central model parameter update
        model.load_state_dict(w_glob)

        print('-' * 20)
        print(str(loss_meter))

        # current model evaluation
        vip_loss, vip_acc = local.evaluate(data_loader=dataloader_viper['val'], model=model)
        print('Current Central Model VIPeR Loss: {:.4f}, Accuracy: {:.4f} '.format(vip_loss,vip_acc))
        writer.add_scalar('baseline/server/VIPeR accuracy'.format(idx),
                    vip_acc,
                    epoch)
        writer.add_scalar('baseline/server/VIPeR loss'.format(idx),
                    vip_loss,
                    epoch)

        val_loss, val_acc = local.evaluate(data_loader=dataloaders_val['val'], model=model)
        print('Current Central Model Validattion Loss: {:.4f}, Validation accuracy: {:.4f}'.format(val_loss,val_acc))
        writer.add_scalar('baseline/server/Validation set accuracy'.format(idx),
                    val_acc,
                    epoch)
        writer.add_scalar('baseline/server/Validation set loss'.format(idx),
                    val_loss,
                    epoch)
        # save the central server model with the best validation loss (the lowest validation loss)
        # Conditional updating in mapping network can lead to low acc but will not affect central model performance
        # as validation can stabilise the central model performance
        if epoch < 0.7*num_epochs: # to stabilise the result, only save the model after 70% training epochs
            with open(model_saved, 'wb') as f:
                torch.save(model.state_dict(), f)
        else:
            if not best_val_loss or val_loss < best_val_loss:
                with open(model_saved, 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_val_loss = val_loss
    # compute training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model
