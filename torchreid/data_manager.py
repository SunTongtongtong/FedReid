from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset, VideoDataset
from .datasets import init_imgreid_dataset, init_vidreid_dataset
from .transforms import build_transforms
from .samplers import RandomIdentitySampler

from args import argument_parser
# global variables
parser = argument_parser()
args = parser.parse_args()

from PIL import Image
import os.path as osp
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataManager(object):

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        try:
            return self.train_loaders, self.testloader_dict, self.finetuneloader
        except:
            return self.train_loaders, self.testloader_dict

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']


class ImageDataManager(BaseDataManager):
    """
    Image-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 finetune_names,
                 root,
                 split_id=0,
                 height=256,
                 width=128,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 num_instances=4, # number of instances per identity (for RandomIdentitySampler)
                 cuhk03_labeled=False, # use cuhk03's labeled or detected images
                 cuhk03_classic_split=False # use cuhk03's classic split or 767/700 split
                 ):
        super(ImageDataManager, self).__init__()
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.finetune_names = finetune_names
        self.root = root
        self.split_id = split_id
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.num_instances = num_instances
        self.cuhk03_labeled = cuhk03_labeled
        self.cuhk03_classic_split = cuhk03_classic_split
        #self.pin_memory = True if self.use_gpu else False
        self.pin_memory = False  #shitong : bacause of limited memory when kd, so set false now.todo 
        global args
        

        # Build train and test transform functions
        transform_train = build_transforms(self.height, self.width, is_train=True)
        transform_test = build_transforms(self.height, self.width, is_train=False)

        print("=> Initializing TRAIN (source) datasets")
        self._num_train_pids = 0
        self._num_train_cams = 0

        #shitong
        self.train_loaders={}
        self.train_dataset_sizes = {}
        self.train_class_sizes = {}
        # shitong add
        if args.kd:
            print('Training with knowledge distillation')

            dataset_kd = init_imgreid_dataset(
                root=self.root, name='cuhk02', split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split
            )
            train_kd=[]
            for img_path, pid, camid in dataset_kd.train:
                train_kd.append((img_path, pid, camid))
            self.kd_loader = DataLoader(
                    ImageDataset(train_kd, transform=transform_train, is_train=True),
                    batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=True
                )

        for name in self.source_names:
            self.train = []
            dataset = init_imgreid_dataset(
                root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split
            )

            for img_path, pid, camid in dataset.train:
                # pid += self._num_train_pids
                # camid += self._num_train_cams   # shitong delete it as not joint training datasets
                self.train.append((img_path, pid, camid))
                # self.train.append((read_image(img_path), img_path, pid, camid)):load all images into memory

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

            if self.train_sampler == 'RandomIdentitySampler':
                trainloader = DataLoader(
                    ImageDataset(self.train, transform=transform_train, is_train=True),
                    sampler=RandomIdentitySampler(self.train, self.train_batch_size, self.num_instances),
                    batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=True
                )

            else:
                trainloader = DataLoader(
                    ImageDataset(self.train, transform=transform_train, is_train=True),
                    batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=True
                )
            self.train_loaders[name] = trainloader
            self.train_dataset_sizes[name] = dataset.num_train_imgs
            self.train_class_sizes[name] = dataset.num_train_pids
        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}
        
        for name in self.target_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split
            )
            if args.use_B92_28id_0604_clip_mode:
                self.testloader_dict[name]['query'] = DataLoader(
                    ImageDataset(dataset.query, transform=transform_test, is_train=False),
                    sampler=RandomIdentitySampler(dataset.query, self.test_batch_size, self.num_instances),
                    batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=False
                )

                self.testloader_dict[name]['gallery'] = DataLoader(
                    ImageDataset(dataset.gallery, transform=transform_test, is_train=False),
                    sampler=RandomIdentitySampler(dataset.gallery, self.test_batch_size, self.num_instances),
                    batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=False
                )
            else:
                self.testloader_dict[name]['query'] = DataLoader(
                    ImageDataset(dataset.query, transform=transform_test, is_train=False),
                    batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=False
                )

                self.testloader_dict[name]['gallery'] = DataLoader(
                    ImageDataset(dataset.gallery, transform=transform_test, is_train=False),
                    batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=False
                )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery
            self.testdataset_dict[name]['train'] = dataset.train

        if args.Unsupervised_meta_finetune:
            print("=> Initializing FINETUNR (traget-finetune) datasets")
            self.finetune = []
            self._num_finetune_pids = 0
            self._num_finetune_cams = 0

            for name in self.finetune_names:
                dataset = init_imgreid_dataset(
                    root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                    cuhk03_classic_split=self.cuhk03_classic_split
                )

                for img_path, pid, camid in dataset.train:
                    pid += self._num_train_pids
                    camid += self._num_train_cams
                    self.finetune.append((read_image(img_path), img_path, pid, camid))

                self._num_finetune_pids += dataset.num_train_pids
                self._num_finetune_cams += dataset.num_train_cams

            if self.train_sampler == 'RandomIdentitySampler':
                self.finetuneloader = DataLoader(
                    ImageDataset(self.finetune, transform=transform_train, is_train=True),
                    sampler=RandomIdentitySampler(self.finetune, self.train_batch_size, self.num_instances),
                    batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=True
                )

            else:
                self.finetuneloader = DataLoader(
                    ImageDataset(self.finetune, transform=transform_train, is_train=True),
                    batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                    pin_memory=self.pin_memory, drop_last=True
                )

        print("\n")
        print("  **************** Summary ****************")
        print("  train names      : {}".format(self.source_names))
        print("  # train datasets : {}".format(len(self.source_names)))
        print("  # train ids      : {}".format(self._num_train_pids))
        print("  # train images   : {}".format(len(self.train)))
        print("  # train cameras  : {}".format(self._num_train_cams))
        print("  test names       : {}".format(self.target_names))
        if args.Unsupervised_meta_finetune:
            print("  finetune names   : {}".format(self.finetune_names))
            print("  # finetune datasets : {}".format(len(self.finetune_names)))
            print("  # finetune ids   : {}".format(self._num_finetune_pids))
            print("  # finetune images   : {}".format(len(self.finetune)))
            print("  # finetune cameras  : {}".format(self._num_finetune_cams))
        print("  *****************************************")

        print("\n")

