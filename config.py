#
# Model training parameter setting
#

import argparse
import os

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str,help='GPU IDs: e.g. 0, 1, 2, ...')
parser.add_argument('--name',default='FedReID', type=str, help='Model Name')
parser.add_argument('--logs_dir', type=str, help='path of logs',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_save'))

# Four local clients
parser.add_argument('--nusers', default=4, type=int, help='number of clients in federated learning')
parser.add_argument('--data_dir_1',type=str, help='path of local client dataset',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'sourceDataset/dukemtmc-reid/pytorch'))
    #default=os.path.join('/import/xxx/dukemtmc-reid/pytorch'))
parser.add_argument('--data_dir_2',type=str, help='path of local client dataset',\
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'sourceDataset/market1501/pytorch'))
parser.add_argument('--data_dir_3',type=str, help='path of local client dataset',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'sourceDataset/msmt17/pytorch'))
parser.add_argument('--data_dir_4',type=str, help='path of local client dataset',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'sourceDataset/cuhk03-np/pytorch'))

# Hyper-parameters
#for testing
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--lr_init', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients: C')
parser.add_argument('--agg', type=str, default='avg', help='Federated average strategy')
parser.add_argument('--dp', type=float, default=0.000, help='differential privacy: beta')
parser.add_argument('--local_bs', type=int, default=32, help="local batch size")
parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs: t_max")
parser.add_argument('--global_ep', type=int, default=100, help="number of global epochs: k_max")
parser.add_argument('--T', type=int, default=3, help="temperature to control the softness of probability distributions")
parser.add_argument('--alpha_mu', type=float, default=0.5, help="update momentum in local weight aggregation: alpha")

#shitong
parser.add_argument('--PBT_mode', dest='PBT',action='store_true', default=False,help="set as PBT mode")
parser.add_argument('--PBT_num', type=int, default=3, help="number of FL block, n client->one block")

#for representation disentangle
parser.add_argument('--disentangle', dest='disEN',action='store_true', default=False,help="set as representation disentangle mode")

#for SNR
parser.add_argument('--CBAM', action='store_true',
                    help="use channel and spatial attention to separate")
parser.add_argument('--CBAM-S', action='store_true',
                    help="only use spatial attention to separate")
parser.add_argument('--C-S-att-parallel', action='store_true',
                    help="use spatial attention and channel attention in parallel to separate")
parser.add_argument('--pool-conv', action='store_true',
                    help="use pooling and 1*1conv embedding to compute distance")
parser.add_argument('--ratio', type=int, default=16,
                    help='control the reduction of channel attention')
parser.add_argument('--label-smooth', action='store_true',
                    help="use label smoothing regularizer in cross entropy loss")
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")

parser.add_argument('--lambda-xent', type=float, default=1,
                    help="weight to balance cross entropy loss")
parser.add_argument('--lambda-htri', type=float, default=1,
                    help="weight to balance hard triplet loss")

parser.add_argument('--lambda-inner', type=float, default=1,
                    help="weight to balance inner hard triplet loss")
parser.add_argument('--layer1-inner', type=float, default=1,
                    help="weight to decide which layer to add inner hard triplet loss")
parser.add_argument('--layer2-inner', type=float, default=1,
                    help="weight to decide which layer to add inner hard triplet loss")
parser.add_argument('--layer3-inner', type=float, default=1,
                    help="weight to decide which layer to add inner hard triplet loss")
parser.add_argument('--layer4-inner', type=float, default=1,
                    help="weight to decide which layer to add inner hard triplet loss")
parser.add_argument('--layer1-sr-mutual', type=float, default=1,
                    help="weight to decide which layer to add inner hard triplet loss")
parser.add_argument('--layer2-sr-mutual', type=float, default=1,
                    help="weight to decide which layer to add inner hard triplet loss")
parser.add_argument('--layer3-sr-mutual', type=float, default=1,
                    help="weight to decide which layer to add inner hard triplet loss")
parser.add_argument('--layer4-sr-mutual', type=float, default=1,
                    help="weight to decide which layer to add inner hard triplet loss")

parser.add_argument('--only-P', action='store_true',
                    help="only use positive to constrain")
parser.add_argument('--only-N', action='store_true',
                    help="only use negative to constrain")
parser.add_argument('--only-Sr', action='store_true',
                    help="don't use the comparison technique to constrain")
parser.add_argument('--only-Sr-type2', action='store_true',
                    help="don't use the comparison technique to constrain, softplus version")
parser.add_argument('--Vis_actMap', action='store_true',
                    help="Visualize feature maps")
parser.add_argument('--Vis_selective_weights', action='store_true',
                    help="Vis_selective_Weights")
parser.add_argument('--Vis_KLMap', action='store_true',
                    help="Visualize KL divergence feature maps")
parser.add_argument('--Vis_tSNEMap', action='store_true',
                    help="Visualize tSNE feature maps")
parser.add_argument('--Vis_final_reid_vector_tSNE', action='store_true',
                    help="Visualize final reid_vector tSNE feature maps")
parser.add_argument('--Vis_hist_threshold', action='store_true',
                    help="Visualize the histogram of TP&TN, and decide the suitable threshold for real-world application")
opt = parser.parse_args()
