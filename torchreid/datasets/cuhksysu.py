import sys
import os
import os.path as osp
import glob
import copy

from .bases import BaseImageDataset

class CUHKSYSU(BaseImageDataset):
    dataset_dir = 'cuhksysu'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(CUHKSYSU, self).__init__()

        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid).
        self.data_dir = osp.join(self.dataset_dir, 'cropped_images')
        # image name format: p11422_s16929_1.jpg
        train = self.process_dir(self.data_dir )
        query = [copy.deepcopy(train[0])]
        gallery = [copy.deepcopy(train[0])]

        if verbose:
            print("=> CUHK-SYSU loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def process_dir(self, dirname):
        img_paths = glob.glob(osp.join(dirname, '*.jpg'))
        num_imgs = len(img_paths)

        # get all identities:
        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = img_name.split('_')[0]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        num_pids = len(pid_container)

        # extract data
        data = []
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = img_name.split('_')[0]
            label = pid2label[pid]
            data.append((img_path, label, 0)) # dummy camera id

        return data