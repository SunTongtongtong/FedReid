# 
# model parameter aggregation in FedReID
#

import copy
import torch
import torch.nn as nn

def weights_aggregate(models, w_glob, dp, alpha_mu, is_local, do_mask, idx_client = [0,1,2,3],client_weights = 0.25):
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

    client_weights = 0.25, average of four clients 
    """
    w_avg = copy.deepcopy(w_glob)
    for key in w_glob.keys():

        if key[0:11] == 'classifier_':
            idx_map = int(key[11])-1
            # aggregate all weights
            for j in range(len(idx_client)):
                if idx_client[j] == idx_map:
                    w_glob[key] = models[j][key]
                    w_avg[key] = models[j][key]

                    break # only update the client participates in the current aggregation

        else: # feature embedding network updating
            temp = torch.zeros_like(w_glob[key], dtype=torch.float32)
            f_mask = torch.ones_like(w_glob[key], dtype=torch.bool)

            for client_idx in range(len(idx_client)):
                if do_mask:
                    delta_idx = models[client_idx][key] - w_glob[key]
                    try: 
                        f_mask = f_mask * (mask == delta_idx.ge(0))
                        mask == delta_idx.ge(0) # can be delete 
                    except:
                        mask = delta_idx.ge(0)

                temp += models[client_idx][key]

            temp = torch.div(temp, len(idx_client))  #shitong todo: change here, add mask
            if do_mask:
                temp = temp*f_mask+ ~f_mask*w_glob[key]
                del mask
                del f_mask
            w_avg[key].data.copy_(temp)     # w_avg means including bn, all the parameters are the average    
            if 'bn' not in key and 'downsample.1' not in key:
    
                w_glob[key].data.copy_(temp)
                for client_idx in range(len(idx_client)):
                    models[client_idx][key].data.copy_(w_glob[key])
            #shitong comment here
                #w_agg[k] = torch.div(w_agg[k], len(w)) + torch.mul(torch.randn(w_agg[k].shape), dp).type_as(w_agg[k])

    
    return w_glob,models,w_avg
