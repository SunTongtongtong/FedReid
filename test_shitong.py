# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from args import argument_parser, image_dataset_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision, TripletLoss_mutual_sr
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers, close_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optimizer
from torchreid.transforms import *
from torch.nn import functional as F

#from utils import get_model, extract_feature
#from utils.utils import extract_feature

from lib.model import embedding_net, embedding_net_test
from utils.load_network import load_network

# from adaBN import bn_update


# global variables
parser = argument_parser()
args = parser.parse_args()
def main():
    global args

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device ',device)

    #if args.use_cpu: use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager to 10 different splits")
    rank1_list=[]
    map_list=[]
    for split_nr in range(10): #shitong todo: change to 10 in small dataset
        args.split_id = split_nr
        dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
        trainloader, testloader_dict = dm.return_dataloaders()

        print("Initializing model:")

        #full_model = get_model(750, args.drop_rate, args.stride).to(device)
        model = embedding_net(702)
        model = embedding_net_test(model)

        model = load_network(model, args.load_weights, args.gpu_devices) # Model restoration from saved model

        model = model.cuda()
        print("args.target_names[0]",args.target_names[0])
        # import pdb
        # pdb.set_trace()
        # print(model.state_dict().keys())
        # bn_update(model, testloader_dict[args.target_names[0]]['gallery'])#,cumulative = not args.adabn_emv)

        model = model.eval()
        # full_model.classifier.classifier = nn.Sequential()
        # model=full_model

        print("Model size: {:.3f} M".format(count_num_param(model)))

        print("Loaded pretrained weights from '{}'".format(args.load_weights))

        if use_gpu:
            model = nn.DataParallel(model).cuda()

        if args.evaluate:
            print("Evaluate only")

            for name in args.target_names:
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                epoch = None

                rank1, map = test(model, queryloader, galleryloader, use_gpu, epoch, return_distmat=True)
                rank1_list.append(rank1)
                map_list.append(map)

    print('======average result on dataset {} ========'.format(args.target_names))
    print('Rank1:',sum(rank1_list)/len(rank1_list))
    print('mAP:',sum(map_list)/len(map_list))
    return

def test(model, queryloader, galleryloader, use_gpu, epoch, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, img_paths) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t()) # (x-y)^2 = x^2+y^2-2xy
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
 
    args.use_metric_cuhk03=True
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
         print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc[0] ,mAP



def compute_accuracy(roc):
    _v = np.array([v for k, v in roc.items()])
    _acc = (_v[:, 0] + _v[:, 2]) / (_v[:, 0] + _v[:, 1] + _v[:, 2] + _v[:, 3])
    return _acc, (_v[:, 0] + _v[:, 2]), (_v[:, 1] + _v[:, 3])


if __name__ == '__main__':
    main()
