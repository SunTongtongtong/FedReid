#
# Extract query and gallery features for evaluation
#

import torch
from torch.autograd import Variable
import pdb

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)

    return img_flip


def eval_feat_ext(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
       # print('Process count:', count)

        ff = torch.FloatTensor(n, 2048).zero_()

        # extract feature for images with horizontal flip
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            # pdb.set_trace()
            f = outputs.data.cpu()
            ff = ff+f

        # normalise feature
        #fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        #ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
        
    return features
