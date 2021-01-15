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

from lib.model import embedding_net
from config import opt
from utils.get_dataset import get_dataset
from lib.fedreid_train import FedReID_train
from lib.fedreid_pbt_train import FedReID_pbt_train
from lib.fedreid_disEN_train import FedReID_disEN_train

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
    local_datasets, dict_users, dataloaders_val, n_data  = get_dataset(opt, is_training=True)

    num_ids_client = [] # number of training ids in each local client
    for i in range(opt.nusers):
        num_ids_client.append(len(local_datasets[i].imagefolder.classes))

    # Model initialisation
    model = embedding_net(num_ids_client)  #list length 4=> will build 4 model here=> actually one model with 4 fully connected layers
                                           # when forward: set an parameter to choose the fully connected layer

    if torch.cuda.is_available():
        model = model.cuda()

    w_glob = model.cpu().state_dict() # weights of neurons
    print('Done')

    # Model training
    print('----------Training----------')
    model = FedReID_train(model, w_glob, opt, local_datasets, dict_users, dataloaders_val,n_data) # Central model

if __name__ == '__main__':
    main(opt)

