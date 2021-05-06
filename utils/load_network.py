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

   # with open(save_path, 'rb') as f:
       # network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['avg_model']) # map the model to the current GPU
    pretrain_dict = torch.load(save_path, map_location='cuda:%s'%gpu_ids[0])['avg_model']
#    import pdb
#    pdb.set_trace()
    model_dict = network.state_dict()
    # import pdb
    # pdb.set_trace()
   # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if 'running_mean' not in k and  'running_var' not in k}
    # import pdb
    # pdb.set_trace()
    model_dict.update(pretrain_dict)
    network.load_state_dict(model_dict)

    #for k, v in pretrain_dict.items():

   


  #  print("Loaded pretrained weights from '{}'".format(args.load_weights))
    # with open(save_path, 'rb') as f:
    #     network1.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['model_1']) # map the model to the current GPU
    # with open(save_path, 'rb') as f:
    #     network2.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['model_2']) # map the model to the current GPU
    # with open(save_path, 'rb') as f:
    #     network3.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['model_3']) # map the model to the current GPU
    # with open(save_path, 'rb') as f:
    #     network_sever.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['server_model']) # map the model to the current GPU
    # import pdb
    # pdb.set_trace()
    # print('Im here')

    return network

    # network0.state_dict().keys()

    # model.layer3.5.bn1.running_mean
    # model.layer1.2.bn3.running_mean
    # model.layer1.0.bn1.num_batches_tracked

    # model.layer1.0.conv2.weight
    # model.layer2.0.downsample.1.running_mean
    # model.layer3.2.conv1.weight

 