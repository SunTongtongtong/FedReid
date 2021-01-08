from __future__ import division, print_function, absolute_import
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import os
from shutil import copyfile


# Log
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'msmt17': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}

class CUHK03(Dataset):

    dataset_dir = 'cuhk03'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        # self.list_train_path = osp.join(
        #     self.dataset_dir, 'list_train.txt'
        # )
        # self.list_val_path = osp.join(
        #     self.dataset_dir, 'list_val.txt'
        # )
        self.list_train_path = osp.join(
            self.dataset_dir, 'train_cuhk03_labeled.txt'
        )
        self.list_query_path = osp.join(
            self.dataset_dir, 'query_cuhk03_labeled.txt'
        )
        self.list_gallery_path = osp.join(
            self.dataset_dir,'gallery_cuhk03_labeled.txt'
        )

        required_files = [self.dataset_dir, self.train_dir, self.test_dir]


        train = self.process_dir( self.list_train_path,'train')
        # val = self.process_dir(self.train_dir, self.list_val_path,'val')
        query = self.process_dir( self.list_query_path,'query')
        gallery = self.process_dir(self.list_gallery_path,'gallery')

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        # if 'combineall' in kwargs and kwargs['combineall']:
        #     train += val

        #super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self,  list_path,type_dataset):
        save_path = osp.join(osp.dirname(list_path),type_dataset)
        if not osp.isdir(save_path):
            os.mkdir(save_path)

        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dir_path = '/homes/ss014/datasets/'
        data = []
        for img_idx, img_info in enumerate(lines):
            im_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(im_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, im_path)
            data.append((img_path, pid, camid))

            src_path = img_path
            aa = im_path.split('/')[1]
            dst_id = str(pid).zfill(4)
            dst_im = osp.join(dst_id,aa)

            dst_path = osp.join(save_path,dst_im)
            if not os.path.isdir(osp.dirname(dst_path)):
                os.mkdir(osp.dirname(dst_path))
            copyfile(src_path,dst_path)
        return data


if __name__ =='__main__':
    root = '/homes/ss014/datasets/cuhk03'
    dataset = CUHK03(root=root)
