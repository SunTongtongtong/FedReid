import argparse
def argument_parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--model_name',default='new_experiment', type=str, help='output model name')
    parser.add_argument('--project_dir',default='.', type=str, help='project path')
    parser.add_argument('--data_dir',default='data',type=str, help='training dir path')
    parser.add_argument('--datasets',default='Market,DukeMTMC-reID,cuhk03-np-detected,cuhk01,MSMT17,viper,prid,3dpes,ilids',type=str, help='datasets used')
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=0.5, type=float, help='drop rate')

    # arguments for federated setting
    parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_of_clients', default=9, type=int, help='number of clients')

    # arguments for data transformation
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )

    # arguments for testing federated model
    parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--multi', action='store_true', help='use multiple query' )
    parser.add_argument('--multiple_scale',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
    parser.add_argument('--test_dir',default='all',type=str, help='./test_data')

    # arguments for optimization
    parser.add_argument('--cdw', action='store_true', help='use cosine distance weight for model aggregation, default false' )
    parser.add_argument('--kd', action='store_true', help='apply knowledge distillation, default false' )
    parser.add_argument('--regularization', action='store_true', help='use regularization during distillation, default false' )

    #change to torchreid data source target

    parser.add_argument('--root', type=str, default='data',
                        help="root path to data directory")
    parser.add_argument('-s', '--source-names', type=str, required=True, nargs='+',
                        help="source datasets (delimited by space)")
    parser.add_argument('-t', '--target-names', type=str, required=True, nargs='+',
                        help="target datasets (delimited by space)")
    parser.add_argument('-f', '--finetune-names', type=str, required=True, nargs='+',
                        help="finetune datasets (delimited by space)")
    parser.add_argument('-j', '--workers', default=12, type=int,
                        help="number of data loading workers (tips: 4 or 8 times number of gpus)")
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image")
    parser.add_argument('--split-id', type=int, default=0,
                        help="split index (note: 0-based)")
    parser.add_argument('--train-sampler', type=str, default='',
                        help="sampler for trainloader")

    parser.add_argument('--train-batch-size', default=64, type=int,
                        help="training batch size")
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help="test batch size")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="number of instances per identity")

    # ************************************************************
    # CUHK03-specific setting
    # ************************************************************
    parser.add_argument('--cuhk03-labeled', action='store_true',
                        help="use labeled images, if false, use detected images")
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                        help="use classic split by Li et al. CVPR'14")
    parser.add_argument('--use-metric-cuhk03', action='store_true',
                        help="use cuhk03's metric for evaluation")

    parser.add_argument('--Not_use_RS', action='store_true',
                        help="Don't use randome erasing operation in the cross-domain training")
    parser.add_argument('--CombineAll', action='store_true',
                        help="CombineAll for MSMT17 datasets, 12w data")
    parser.add_argument('--CombineAll4Duke', action='store_true',
                        help="CombineAll for Duke datasets, 3.5w data")
    parser.add_argument('--CombineAll4Market', action='store_true',
                        help="CombineAll for Market1501 datasets, 3.2w data")

    parser.add_argument('--use_B92_28id_0604_clip_mode', action='store_true',
                        help="use OMG dataset for evaluation, need use 'RandomidentitySampler' for sample in clips")
    parser.add_argument('--Unsupervised_meta_finetune', action='store_true',
                        help="Enable unsupervised meta-learner based finetune strategy")

#shitong add for testing
    parser.add_argument('--load-weights', type=str, default='',
                        help="load pretrained weights but ignore layers that don't match in size")
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=1,
                        help="manual seed")
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help="use available gpus instead of specified devices (useful when using managed clusters)")
    parser.add_argument('--use-cpu', action='store_true',
                        help="use cpu")
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluate only")

    parser.add_argument('--save-dir', type=str, default='log',
                        help="path to save log and model weights")

    return parser

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
        'cuhk03_labeled': parsed_args.cuhk03_labeled,
        'cuhk03_classic_split': parsed_args.cuhk03_classic_split
    }