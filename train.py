# 
# Decentralised Person Re-Identification
# Guile Wu and Shaogang Gong
# 2020
#
# I can 
from __future__ import print_function, division
import torch
import numpy as np
import time

from lib.model import embedding_net,embedding_net_test
from config import opt
from utils.get_dataset import get_dataset
from lib.fedreid_train import FedReID_train



def main(opt):
    # Set GPU

    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)

    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])

    print('Using GPU {} for model training'.format(gpu_ids[0]))

    # Prepare local client datasets
    print('----------Load client datasets----------')
    local_datasets, dict_users, dataloaders_val,dataloader_viper  = get_dataset(opt, is_training=True)

    num_ids_client = [] # number of training ids in each local client
    for i in range(opt.nusers):
        num_ids_client.append(len(local_datasets[i].classes))

    # Model initialisation

    models = []
    for ids in num_ids_client:        
        model_idx = embedding_net(ids).cuda()  #list length 4=> will build 4 model here=> actually one model with 4 fully connected layers
        models.append(model_idx)                                   # when forward: set an parameter to choose the fully connected layer

    glob_model = embedding_net(30).cuda()
    w_glob = embedding_net_test(glob_model).state_dict() # weights of neurons
    print('Done')

    # Model training
    print('----------Training----------')
    model = FedReID_train(models, w_glob, opt, local_datasets, dict_users, dataloaders_val,dataloader_viper) # Central model

if __name__ == '__main__':
    main(opt)

