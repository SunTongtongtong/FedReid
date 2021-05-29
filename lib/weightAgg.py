# 
# model parameter aggregation in FedReID
#

import copy
import torch

def weights_aggregate(w_lg,w_ls,w_glob,idx_client):
    """
    w: client model parameters
    dp: differential privacy scale factor beta
    alpha_mu: update momentum in local weight aggregation alpha
    idx_client: selected local client index
    is_local: flag of local client
    return: aggregated model parameters

    shitong
    w[0]->global
    w[1]->local->expert
    """

    for k in w_glob.keys(): 
        
            # central server use the average of selected local clients for aggregation
        temp = torch.zeros_like(w_glob[k], dtype=torch.float32)
        # conv layers should from w_ls
        if 'bn' not in k and 'downsample.1' not in k: 
        # if 'bn' not in k and 'layer1.0.downsample.1' not in k  \
        # and 'conv1.1' not in k and 'conv1.4' not in k and 'downsample.2' not in k:
            for i in range(len(idx_client)):
                temp += w_ls[idx_client[i]][k]        
        #bn layers should from w_lg
        else:
            for i in range(len(idx_client)):
                temp += w_lg[idx_client[i]][k]
        # privacy protection with differential privacy                                      
        temp = torch.div(temp, len(idx_client))
        w_glob[k].data.copy_(temp)   
        if 'bn' not in k and 'downsample.1' not in k:   
        # if 'bn' not in k and 'layer1.0.downsample.1' not in k  \
        #     and 'conv1.1' not in k and 'conv1.4' not in k and 'downsample.2' not in k:
            for i in range(len(idx_client)):
                w_ls[idx_client[i]][k].data.copy_(temp)
                w_lg[idx_client[i]][k].data.copy_(temp)
        else:
            for i in range(len(idx_client)):
                w_lg[idx_client[i]][k].data.copy_(temp)        
    return w_lg,w_ls,w_glob


