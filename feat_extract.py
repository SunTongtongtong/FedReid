#
# Extract Feature Representation of Testing Data
#

from __future__ import print_function, division
import torch
import scipy.io


from utils.get_dataset import get_dataset
from utils.load_network import load_network

from config_test import opt
from lib.model import embedding_net, embedding_net_test

from evaluation.get_id import get_id, get_id_cuhk_msmt
from evaluation.eval_feat_ext import eval_feat_ext#, fliplr
from lib.model import embedding_net
import torch.optim as optim


from tent import tent

def main(opt):

    # Set GPU  
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            gpu_ids.append(id)

    # Set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])

    use_gpu = torch.cuda.is_available()

    # Load testing data
    print('----------Load testing data----------')
    image_datasets, dataloaders  = get_dataset(opt, is_training=False)
    print('Done.')

    # Get camera and identity labels of gallery and query
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    # gallery_cam, gallery_label = get_id(gallery_path)
    # query_cam, query_label = get_id(query_path)

    if gallery_path[0][0].split('/')[-5] not in ['msmt17', 'cuhk03-np']:
        gallery_cam, gallery_label = get_id(gallery_path)
        query_cam, query_label = get_id(query_path)
    else:
        gallery_cam, gallery_label = get_id_cuhk_msmt(gallery_path)
        query_cam, query_label = get_id_cuhk_msmt(query_path)

    print('----------Extracting features----------')
    # Model initialisation, we use four local client in current version (Duke, Market, MSMT, CUHK03)
    # model = embedding_net([702, 751, 1041, 767])
    model = embedding_net([702, 751, 1041, 767])
    model = load_network(model, opt.model_name, gpu_ids) # Model restoration from saved model
    # Remove the mapping network and set to embedding feature extraction
    model = embedding_net_test(model)

    # Change to test mode
    model = model.eval()

    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = optim.Adam(params, lr=1e-3,betas=(0.9, 0.999),weight_decay=0.0)
    tented_model = tent.Tent(model, optimizer)
    model = tented_model
    

    print('Im tented model')

    if use_gpu:
        model = model.cuda()

    # Extract feature
    model.is_query = False
    gallery_feature = eval_feat_ext(model, dataloaders['gallery'])
    print('Done gallery.')
    model.is_query = True

    query_feature = eval_feat_ext(model, dataloaders['query'])
    print('Done query.')

    print('----------Saving features----------')
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,
              'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('result_feature.mat',result)
    print('Done.')


if __name__ == '__main__':
    main(opt)


