# 
# model parameter aggregation in FedReID
#

import copy
import torch

def weights_aggregate(w_all,w_glob, idx_client):
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
            # low kl loss-> low model weight
        temp = torch.zeros_like(w_glob[k], dtype=torch.float32)
        for i in range(len(w_all)):
            temp += w_all[i][k]
            # w_agg[k] = w_agg[k] + w_all[i][k] #* kl_weight[i]
        # privacy protection with differential privacy
        temp = torch.div(temp, len(idx_client))
        w_glob[k].data.copy_(temp)   
        for i in range(len(w_all)):
            w_all[i][k].data.copy_(temp)
        #shitong comment here
        # w_agg[k] = torch.div(w_agg[k], len(w_all)) + torch.mul(torch.randn(w_agg[k].shape), dp).type_as(w_agg[k])
    return w_glob,w_all


