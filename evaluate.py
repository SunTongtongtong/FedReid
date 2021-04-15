#
# Evaluating with saved features
#

import scipy.io
import torch
import numpy as np
import os
from evaluation.compute_acc import compute_acc

# set gpu device
torch.cuda.set_device(0)

# load features and labels
result = scipy.io.loadmat('result_feature.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

# compute rank accuracy and mAP
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
for i in range(len(query_label)):
   # print('Process count:', i)
    ap_tmp, CMC_tmp = compute_acc(query_feature[i],query_label[i],query_cam[i],
                                    gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9], ap/len(query_label)))
