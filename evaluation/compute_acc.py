#
# Compute rank accuracy and mAP based on euclidean/L2 distance score matrix
#

import torch
import numpy as np
from evaluation.compute_mAP import compute_mAP

def compute_acc(qf,ql,qc,gf,gl,gc):
    # compute euclidean/L2 distance score matrix
    qf = qf.view(1,-1)
    m, n = 1, gf.size(0)
    qq = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
    gg = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m)
    l2_q_g = qq + gg.t()
    l2_q_g.addmm_(1, -2, qf, gf.t())
    #comment or uncomment the following line will not affect the result, but can save computational time
    l2_q_g = l2_q_g.clamp(min=1e-12).sqrt() #for numerical stability
    l2_q_g = l2_q_g.cpu()
    l2_q_g = l2_q_g.numpy()

    l2_q_g_index = np.argsort(l2_q_g) #ascending order
    index = l2_q_g_index.reshape(-1, 1)

    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

    # junk index
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    # compute rank accuracy and mAP
    CMC_tmp = compute_mAP(index, good_index, junk_index)

    return CMC_tmp
