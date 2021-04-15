from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
#from scipy.misc import imsave
import random

from torchreid.utils.iotools import mkdir_if_missing, write_json, read_json
from .bases import BaseImageDataset


# class PRID2011(BaseVideoDataset):
#     """
#     PRID2011
#
#     Reference:
#     Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
#
#     URL: https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/
#
#     Dataset statistics:
#     # identities: 200
#     # tracklets: 400
#     # cameras: 2
#     """
#     dataset_dir = 'prid2011'
#
#     def __init__(self, root='data', split_id=0, min_seq_len=0, verbose=True, **kwargs):
#         self.dataset_dir = osp.join(root, self.dataset_dir)
#         self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')
#         self.cam_a_path = osp.join(self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_a')
#         self.cam_b_path = osp.join(self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_b')
#
#         self._check_before_run()
#         splits = read_json(self.split_path)
#         if split_id >=  len(splits):
#             raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
#         split = splits[split_id]
#         train_dirs, test_dirs = split['train'], split['test']
#         print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))
#
#         train = self._process_data(train_dirs, cam1=True, cam2=True)
#         query = self._process_data(test_dirs, cam1=True, cam2=False)
#         gallery = self._process_data(test_dirs, cam1=False, cam2=True)
#
#         if verbose:
#             print("=> PRID2011 loaded")
#             self.print_dataset_statistics(train, query, gallery)
#
#         self.train = train
#         self.query = query
#         self.gallery = gallery
#
#         self.num_train_pids, _, self.num_train_cams = self.get_videodata_info(self.train)
#         self.num_query_pids, _, self.num_query_cams = self.get_videodata_info(self.query)
#         self.num_gallery_pids, _, self.num_gallery_cams = self.get_videodata_info(self.gallery)
#
#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self.dataset_dir):
#             raise RuntimeError("'{}' is not available".format(self.dataset_dir))
#
#     def _process_data(self, dirnames, cam1=True, cam2=True):
#         tracklets = []
#         dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
#
#         for dirname in dirnames:
#             if cam1:
#                 person_dir = osp.join(self.cam_a_path, dirname)
#                 img_names = glob.glob(osp.join(person_dir, '*.png'))
#                 assert len(img_names) > 0
#                 img_names = tuple(img_names)
#                 pid = dirname2pid[dirname]
#                 tracklets.append((img_names, pid, 0))
#
#             if cam2:
#                 person_dir = osp.join(self.cam_b_path, dirname)
#                 img_names = glob.glob(osp.join(person_dir, '*.png'))
#                 assert len(img_names) > 0
#                 img_names = tuple(img_names)
#                 pid = dirname2pid[dirname]
#                 tracklets.append((img_names, pid, 1))
#
#         return tracklets

class PRID2011(BaseImageDataset):
    """PRID (single-shot version of prid-2011)
    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and Discriminative
        Classification. SCIA 2011.
    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_

    Dataset statistics:
        - Two views.
        - View A captures 385 identities.
        - View B captures 749 identities.
        - 200 identities appear in both views.
    """

    dataset_dir = 'PRID2011'
    dataset_url = None

    def __init__(self, root='data', split_id=0, min_seq_len=0, verbose=True, **kwargs):
        super(PRID2011, self).__init__()

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.cam_a_dir = osp.join(self.dataset_dir, 'single_shot', 'cam_a')
        self.cam_b_dir = osp.join(self.dataset_dir,  'single_shot', 'cam_b')
        self.split_path = osp.join(self.dataset_dir, 'splits_single_shot.json')


        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'.format(split_id, len(splits) - 1))
        split = splits[split_id]

        train, query, gallery = self.process_split(split)

        self.train = train
        self.query = query
        self.gallery = gallery

        #shitong add
        if verbose:
            print("=> PRID2011 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating splits ...')

            splits = []
            for _ in range(10):
                # randomly sample 100 IDs for train and use the rest 100 IDs for test
                # (note: there are only 200 IDs appearing in both views)
                pids = [i for i in range(1, 201)]
                train_pids = random.sample(pids, 100)
                train_pids.sort()
                test_pids = [i for i in pids if i not in train_pids]
                split = {'train': train_pids, 'test': test_pids}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    def process_split(self, split):
        train_pids = split['train']
        test_pids = split['test']

        train_pid2label = {pid: label for label, pid in enumerate(train_pids)}

        # train
        train = []
        for pid in train_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            pid = train_pid2label[pid]
            img_a_path = osp.join(self.cam_a_dir, img_name)
            train.append((img_a_path, pid, 0))
            img_b_path = osp.join(self.cam_b_dir, img_name)
            train.append((img_b_path, pid, 1))

        # query and gallery
        query, gallery = [], []
        for pid in test_pids:
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_a_path = osp.join(self.cam_a_dir, img_name)
            query.append((img_a_path, pid, 0))
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
        for pid in range(201, 750):
            img_name = 'person_' + str(pid).zfill(4) + '.png'
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))

        return train, query, gallery