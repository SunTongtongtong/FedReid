# 
# local client datasets loading and partition
#

import torch
from torchvision import datasets, transforms
import os
from utils.sampling import partition
import pdb

from torch.utils.data import dataset
import numpy as np


class myImageFolder(dataset.Dataset):

    def __init__(self, path, transformer,k=4096):
        self.imagefolder = datasets.ImageFolder(path,transformer)
        num_classes = len(self.imagefolder.classes)
        num_samples = len(self.imagefolder.imgs)
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[self.imagefolder.targets[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        self.mode = 'relax'  # relax means any positive pair ok, otherwise only the same image are positive pair
        self.k = k
        # pdb.set_trace()


    def __getitem__(self, index):
        sample,target = self.imagefolder.__getitem__(index)

        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)

        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return sample, target, index, sample_idx
        #pdb.set_trace()

# Loading data with torchvision and torch.utils.data packages
def get_dataset(opt, is_training=True):
    # load training data
    if is_training:
        # Data augmentation
        transform_train_list = [
                transforms.Resize((288,144), interpolation=3),
                transforms.RandomCrop((256,128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]


        transform_val_list = [
                transforms.Resize(size=(256,128),interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        data_transforms = {
            'train': transforms.Compose( transform_train_list ),
            'val': transforms.Compose(transform_val_list),
        }

        # Local client training data
        local_datasets = []
        #local_datasets.append(datasets.ImageFolder(os.path.join(opt.data_dir_1, 'train'), data_transforms['train']))

        local_datasets.append(myImageFolder(os.path.join(opt.data_dir_1, 'train'), data_transforms['train']))


        local_datasets.append(myImageFolder(os.path.join(opt.data_dir_2, 'train'), data_transforms['train']))
        local_datasets.append(myImageFolder(os.path.join(opt.data_dir_3, 'train'), data_transforms['train']))
        local_datasets.append(myImageFolder(os.path.join(opt.data_dir_4, 'train'), data_transforms['train']))

        # To stabilise model performance, select one client as the validation client
        # and stored the server model with the best performance
        local_datasets_val = {}
        dataloaders_val = {'val': torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(opt.data_dir_1, 'val'),
            data_transforms['val']), batch_size=opt.batchsize, shuffle=True, num_workers=8)}

        # Federated client partition (each client can be partitioned into multiple local users)
        # Here, one user in each client
        dict_users = []
        dict_users.append(partition(len_dataset=len(local_datasets[0]), num_users=1))
        dict_users.append(partition(len_dataset=len(local_datasets[1]), num_users=1))
        dict_users.append(partition(len_dataset=len(local_datasets[2]), num_users=1))
        dict_users.append(partition(len_dataset=len(local_datasets[3]), num_users=1))

        return local_datasets, dict_users, dataloaders_val

    # load testing data
    else:
        data_transforms = transforms.Compose([
                transforms.Resize((288,144), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        image_datasets = {x: datasets.ImageFolder( os.path.join(opt.test_data_dir, x), data_transforms) for x in ['gallery','query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                 shuffle=False, num_workers=16) for x in ['gallery','query']}

        return image_datasets, dataloaders

