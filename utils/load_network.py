#
# Load model
#

import torch
import os
import copy


def load_network(network, save_path, gpu_ids):
    # model_pth = '{}'.format(opt.model_name) # saved model
    # name = opt.name
    # dir_name = os.path.join(opt.logs_dir, name)
    # save_path = os.path.join(dir_name, model_pth)
    # network0 = copy.deepcopy(network)
    # network1 = copy.deepcopy(network)
    # network2 = copy.deepcopy(network)
    # network3 = copy.deepcopy(network)
    # network_sever = copy.deepcopy(network)

    with open(save_path, 'rb') as f:
        network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['avg_model']) # map the model to the current GPU
    # with open(save_path, 'rb') as f:
    #     network0.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['model_0']) # map the model to the current GPU
    # with open(save_path, 'rb') as f:
    #     network1.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['model_1']) # map the model to the current GPU
    # with open(save_path, 'rb') as f:
    #     network2.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['model_2']) # map the model to the current GPU
    # with open(save_path, 'rb') as f:
    #     network3.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['model_3']) # map the model to the current GPU
    import pdb
    pdb.set_trace()
    print('Im here')

    #return network0,network1,network2,network3,network
    return network

    # network0.state_dict().keys()

    # model.layer3.5.bn1.running_mean
    # model.layer1.2.bn3.running_mean
    # model.layer1.0.bn1.num_batches_tracked

    # model.layer1.0.conv2.weight
    # model.layer2.0.downsample.1.running_mean
    # model.layer3.2.conv1.weight

 