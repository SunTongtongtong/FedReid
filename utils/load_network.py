#
# Load model
#

import torch
import os

def load_network(network, opt, gpu_ids):
    model_pth = 'model_{}_{}.pth'.format(opt.agg, opt.frac) # saved model
    name = opt.name
    dir_name = os.path.join(opt.logs_dir, name)
    save_path = os.path.join(dir_name, model_pth)
    with open(save_path, 'rb') as f:
        network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])) # map the model to the current GPU
    return network
 