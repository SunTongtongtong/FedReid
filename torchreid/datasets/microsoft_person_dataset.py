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

from .bases import BaseImageDataset

from args import argument_parser

# global variables
parser = argument_parser()
args = parser.parse_args()


class Microsoft_person_data(BaseImageDataset):

    dataset_dir = 'b92_28id_0603'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(Microsoft_person_data, self).__init__()
        self.dataset_dir = '/home/v-lew/jinx/OMG/B92_28id_0603/'
        self.train_dir = '/mnt/Microsoft_person_data_detection_results/'
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir_ori(self.train_dir, relabel=True)
        query = self._process_dir_query(self.query_dir, relabel=False)
        gallery = self._process_dir_gallery(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Microsoft_person_data loaded, train data come from Microsoft_person_data, query and gallery data no meanings")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_query(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        dataset = []
        for img_path in img_paths:
            pid = int(img_path.split('_')[-1].split('.')[0])
            if 0 <= pid <=10:
                pid = pid - 1
            elif 12 <= pid <=18:
                pid = pid - 2
            elif pid > 18:
                pid = pid -3
            assert 0 <= pid <= 26
            dataset.append((img_path, pid, 0))

        return dataset

    def _process_dir_gallery(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        dataset = []
        for img_path in img_paths:
            pid = int(img_path.split('_')[-1].split('.')[0])
            if 0 <= pid <=10:
                pid = pid - 1
            elif 12 <= pid <=18:
                pid = pid - 2
            elif pid > 18:
                pid = pid -3
            assert 0 <= pid <= 26
            dataset.append((img_path, pid, 1))

        return dataset

    def _process_dir_ori(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        dataset = []
        for img_path in img_paths:
            img_path_pid_cam = img_path.split('/')[-1]
            pid, camid = int(img_path_pid_cam.split('_')[1]), int(img_path_pid_cam.split('_')[3])
            dataset.append((img_path, pid, camid))

        return dataset