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
# from torch.utils.tensorboard import SummaryWriter
from utils.meters import AverageMeter




def FedReID_train(models, w_glob, opt, local_datasets, dict_users, dataloaders_val,dataloader_viper):
    # Model save directory
   
    # writer = SummaryWriter('runs/{}'.format(opt.name) + time.strftime(".%m_%d_%H:%M:%S"))

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

    w_all = []#, w_tmp = [], [] # all local model parameters and local temporal model parameters
    for i in range(opt.nusers):
        w_all.append(models[i].state_dict()) # initial


    loss_meter = AverageMeter('Total loss for selected clients', ':6.3f')

    # Training
    for epoch in range(num_epochs):
        print('Global Training Epoch {}/{}'.format(epoch+1, num_epochs))

        loss_meter.reset()

        idxs_users_selected = np.random.choice(range(opt.nusers), m, replace=False) # randomly selected clients

        # local client model updating
        for idx in idxs_users_selected:
            print('=====training global round {} for client {} =============='.format(epoch,idx))
            # local client model initialisation and local dataset partition
            # idxs: each local client only contains one user here
            local = LocalUpdateLM(args=opt, dataset=local_datasets[idx], idxs=dict_users[idx][0], nround=epoch, user=idx)

            # local client weight update
   
            models[idx].load_state_dict(w_all[idx])

            # local client model training, return model parameters and training loss
            out_dict = local.update_weights(model=copy.deepcopy(models[idx]), cur_epoch=epoch,idx_client=idx)

            # store updated local client parameters
            loss_meter.update(out_dict['loss_meter'])
            #shitong

            # store all local client parameters (some clients are not updated in the randomly selection)
            w_all[idx] = copy.deepcopy(out_dict['params'])
            # writer.add_scalar('baseline/client {} total loss'.format(idx),
            #                   out_dict['loss_meter'],
            #                   epoch)
           
            # writer.add_scalar('baseline/client {} accuracy'.format(idx),
                            #   out_dict['acc'],
                            #   epoch)

        # central server model updating 
        if opt.agg == 'avg': # current version  only supports modified federated average strategy
            w_glob,w_all = weights_aggregate(w_all, w_glob, idx_client=idxs_users_selected)#  central model parameter update
        models[0].load_state_dict(w_all[0]) #shitong want to remove this line

        print('-' * 20)
        print(str(loss_meter))

        # current model evaluation
        val_loss, val_acc = local.evaluate(data_loader=dataloaders_val['val'], model=models[0])
        print('Current Central Model Validation Loss: {:.4f}, Validation accuracy: {:.4f}'.format(val_loss,val_acc))

        if epoch == num_epochs-1: # to stabilise the result, only save the model after 70% training epochs

            with open(model_saved, 'wb') as f:
                # torch.save(w_glob, f)
                torch.save({
                'model_0': w_all[0],
                'model_1': w_all[1],
                'model_2': w_all[2],
                'model_3': w_all[3],
                'server_model': w_glob,
                 }, f)
    # compute training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return models
