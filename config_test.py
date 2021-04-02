#
# Model testing parameter setting
#

import argparse
import os

######################################################################
# Options
# --------
# Hyper-parameters (consistent with training setting)
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_data_dir',type=str, help='path of testing dataset',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'targetDataset/QMUL-iLIDS/pytorch'))
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
#parser.add_argument('--frac', type=float, default=0.5, help='the fraction of clients: C')
#shitong used for test on localized dataset
parser.add_argument('--frac', type=str, default='epoch100', help='the fraction of clients: C')

parser.add_argument('--agg', type=str, default='avg', help='Federated average strategy')
parser.add_argument('--name',default='FedReID', type=str, help='Model Name')
parser.add_argument('--logs_dir', type=str, help='path of logs',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_save'))

#shitong
parser.add_argument('--model_name', type=str, help='model name end with pth',default='model_CRD2048.pth')

opt = parser.parse_args()
