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
parser.add_argument('--name',default='withoutExpert_2nd', type=str, help='Model Name')
parser.add_argument('--logs_dir', type=str, help='path of logs',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_save'))

parser.add_argument('--model_name',default='model_SNR.03_13_21:59:10.pth', type=str, help='Model Name')
#parser.add_argument('--model_path',default='blank', type=str, help='Model Name')

#for test_split.py
# parser.add_argument('--root', type=str, default='data',help="root path to data directory")
# parser.add_argument('-s', '--source-names', type=str, required=True, nargs='+',help="source datasets (delimited by space)")
# parser.add_argument('-t', '--target-names', type=str, required=True, nargs='+',help="target datasets (delimited by space)")
# parser.add_argument('--load-weights', type=str, default='',
#                         help="load pretrained weights but ignore layers that don't match in size")
# parser.add_argument('--evaluate', action='store_true',
#                         help="evaluate only")
# parser.add_argument('--seed', type=int, default=1,
#                         help="manual seed")

# parser.add_argument('--height', type=int, default=256,
#                     help="height of an image")
# parser.add_argument('--width', type=int, default=128,
#                         help="width of an image")

# parser.add_argument('--train-batch-size', default=64, type=int,
#             help="training batch size")
# parser.add_argument('--test-batch-size', default=100, type=int,
#             help="test batch size")
# parser.add_argument('--num-instances', type=int, default=4,
#             help="number of instances per identity")
# parser.add_argument('-j', '--workers', default=12, type=int,
#                         help="number of data loading workers (tips: 4 or 8 times number of gpus)")

# parser.add_argument('--train-sampler', type=str, default='',
#                         help="sampler for trainloader")
# parser.add_argument('-f', '--finetune-names', type=str, required=True, nargs='+',
#                         help="finetune datasets (delimited by space)")

opt = parser.parse_args()


def image_dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'source_names': parsed_args.source_names,
        'target_names': parsed_args.target_names,
        'finetune_names': parsed_args.finetune_names,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'num_instances': parsed_args.num_instances,
        # 'cuhk03_labeled': parsed_args.cuhk03_labeled,
        # 'cuhk03_classic_split': parsed_args.cuhk03_classic_split
    }