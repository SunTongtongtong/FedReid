# 
# model parameter aggregation in FedReID
#

import copy
import torch

def weights_aggregate(w, dp, alpha_mu, idx_client, is_local,kl_loss_locals=[]):
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

    if len(kl_loss_locals)==0:
        w_agg = copy.deepcopy(w[0]) # initial the aggregated parameters
    else:
        kl_weight = [sum(kl_loss_locals) / i for i in kl_loss_locals]
        kl_weight = [i / sum(kl_weight) for i in kl_weight]
        w_agg = copy.deepcopy(w[0])*kl_weight[0]
    for k in w_agg.keys():
        # local client updating
        if is_local:
            # mapping network updating
            # ω can be consecutively update which helps to stabilise the domain-specific optimisation
            # Or, the latest ω can be restored only when the client participates in the last central aggregation
            if k[0:11] == 'classifier_':
                idx_map = int(k[11])-1
                w_agg[k] = w[1][k] if idx_client == idx_map else w_agg[k] #consecutive
                # w_agg[k] = w_agg[k] if idx_client == idx_map else w[1][k] #non-consecutive
                    
            else: # feature embedding network updating
                w_agg[k] = (1-alpha_mu) * w_agg[k] + alpha_mu * w[1][k]
                # privacy protection with differential privacy
                w_agg[k] = w_agg[k] + torch.mul(torch.randn(w_agg[k].shape), dp).type_as(w_agg[k])                

        else: # central server updating
            # mapping network updating
            if k[0:11] == 'classifier_':
                idx_map = int(k[11])-1
                # aggregate all weights
                for j in range(len(idx_client)):
                    if idx_client[j] == idx_map:
                        w_agg[k] = w[j][k]
                        break # only update the client participates in the current aggregation

            else: # feature embedding network updating
                # central server use the average of selected local clients for aggregation
                # low kl loss-> low model weight
                for i in range(1, len(w)):
                    w_agg[k] = w_agg[k] + w[i][k] #* kl_weight[i]
                # privacy protection with differential privacy

                #shitong comment here
                w_agg[k] = torch.div(w_agg[k], len(w)) + torch.mul(torch.randn(w_agg[k].shape), dp).type_as(w_agg[k])


    return w_agg



def weights_disEN_aggregate(w, dp, alpha_mu, idx_client, is_local,global_model=None):
    """
    w: client model parameters
    dp: differential privacy scale factor beta
    alpha_mu: update momentum in local weight aggregation alpha
    idx_client: selected local client index
    is_local: flag of local client
    return: aggregated model parameters

    shitong
    is local
    w[0]->global
    w[1]->local->expert

    is local false
    w 4 elements
    all expert model
    """
    if is_local:
        w_agg = copy.deepcopy(w[1])  # initial the aggregated parameters
    else:
        w_agg = copy.deepcopy(global_model)
    for k in w_agg.keys():
        # local client updating
        if is_local:
            # mapping network updating
            # ω can be consecutively update which helps to stabilise the domain-specific optimisation
            # Or, the latest ω can be restored only when the client participates in the last central aggregation
            if k[0:11] == 'classifier_':
                idx_map = int(k[11]) - 1
                #w_agg[k] = w[1][k] if idx_client == idx_map else w_agg[k]  # consecutive
                # w_agg[k] = w_agg[k] if idx_client == idx_map else w[1][k] #non-consecutive
                w_agg[k] = w[1][k]


            else:  # feature embedding network updating
                w_agg[k] = (1 - alpha_mu) * w_agg[k] + alpha_mu * w[1][k]
                # privacy protection with differential privacy
                w_agg[k] = w_agg[k] + torch.mul(torch.randn(w_agg[k].shape), dp).type_as(w_agg[k])

        else:  # central server updating
            # mapping network updating
            if k[0:11] == 'classifier_':
                idx_map = int(k[11]) - 1
                # aggregate all weights
                for j in range(len(idx_client)):
                    if idx_client[j] == idx_map:
                        w_agg[k] = w[j][k]
                        break  # only update the client participates in the current aggregation

            else:  # feature embedding network updating
                # central server use the average of selected local clients for aggregation
                # low kl loss-> low model weight
                for i in range(1, len(w)):
                    w_agg[k] = w_agg[k] + w[i][k]  # * kl_weight[i]
                # privacy protection with differential privacy

                # shitong comment here
                w_agg[k] = torch.div(w_agg[k], len(w)) + torch.mul(torch.randn(w_agg[k].shape), dp).type_as(w_agg[k])

    return w_agg
