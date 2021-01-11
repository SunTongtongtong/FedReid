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

#for CRDLoss
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mode', default='relax', type=str, choices=['exact', 'relax'])
parser.add_argument('--nce_k', default=4096, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
parser.add_argument('--s_dim', default=2048, type=int, help='student dimension')
parser.add_argument('--t_dim', default=2048, type=int, help='teacher dimension')

opt = parser.parse_args()
