#
# Load model
#

import torch
import os

def load_network(network, save_path, gpu_ids,name):
    # model_pth = '{}'.format(opt.model_name) # saved model
    # name = opt.name
    # dir_name = os.path.join(opt.logs_dir, name)
    # save_path = os.path.join(dir_name, model_pth)
    name_model_dict = {
        'dukemtmc-reid':'model_0',
        'market1501':'model_1',
        'msmt17':'model_2',
        'cuhk03-np':'model_3'
        }    
    if name in name_model_dict.keys():
        with open(save_path, 'rb') as f:
            network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])[name_model_dict[name]]) # map the model to the current GPU
        print('loading network weight from {}'.format(name_model_dict[name]))
        return network
    else:
        with open(save_path, 'rb') as f:
            network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['server_model']) # map the model to the current GPU
        # print('loading network weight from server model') # map the model to the curren    t GPU   
        import pdb
        pdb.set_trace()
        return network
 
