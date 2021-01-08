# 
# FedReID model: feature embedding network + mapping network
#

import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import torch

# Weight initialisation
def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Multi-Layer Perceptron (MLP) as a mapping network
class MLP(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(MLP, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class MLP_disentangle_exp(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(MLP_disentangle_exp, self).__init__()

        add_block=[] #global representation
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.add_block = add_block

        add_block_local=[] #global representation
        add_block_local += [nn.Linear(input_dim, num_bottleneck)]
        add_block_local += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block_local += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block_local += [nn.Dropout(p=0.5)]
        add_block_local = nn.Sequential(*add_block_local)
        add_block_local.apply(weights_init_kaiming)
        self.add_block_local = add_block_local

        classifier = []
        classifier += [nn.Linear(num_bottleneck*2, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        local_r = self.add_block_local(x)
        global_r = self.add_block(x)

        # local_r = self.dropout(self.relu(self.bn1(self.fc1(x))))
        # global_r = self.dropout(self.relu(self.bn2(self.fc2(x))))

        # change to concat the entangled representation instead of sum of them
        x = torch.cat((local_r,global_r),1)
        #x = local_r + global_r
        x = self.classifier(x)
        return x, global_r


    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

class MLP_disentangle_glob(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(MLP_disentangle_glob, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        # if relu:
        #     add_block += [nn.LeakyReLU(0.1)]
        # if dropout:
        #     add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        # classifier = []
        # classifier += [nn.Linear(num_bottleneck, class_num)]
        # classifier = nn.Sequential(*classifier)
        # classifier.apply(weights_init_classifier)

        self.add_block = add_block
        # self.classifier = classifier

    def forward(self, x):
        # print(x.shape)
        middle = self.add_block(x)
        # x = self.classifier(middle)
        return middle


# Feature embedding network 
class embedding_net(nn.Module):
    def __init__(self, num_ids_client, feat_dim=2048):
        super(embedding_net, self).__init__()
        model_backbone = models.resnet50(pretrained=True)
        model_backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_backbone
        self.num_client = len(num_ids_client)

        self.classifier = []
        for i in range(len(num_ids_client)):
            name = 'classifier_'+str(i+1)
            setattr(self, name, MLP(feat_dim, num_ids_client[i]))

    def forward(self, x, idx_client=0):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        assert idx_client <= self.num_client-1
        client_name = 'classifier_' + str(idx_client+1)
        mapping_net = getattr(self, client_name)
        x = mapping_net(x)

        return x


# Feature embedding network
class embedding_disEN_net(nn.Module):
    def __init__(self, num_ids_client, feat_dim=2048):
        super(embedding_disEN_net, self).__init__()
        model_backbone = models.resnet50(pretrained=True)
        model_backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_backbone
        self.num_client = len(num_ids_client)

        self.classifier = []
        for i in range(len(num_ids_client)):
            name = 'classifier_'+str(i+1)
            setattr(self, name, MLP_disentangle_exp(feat_dim, num_ids_client[i]))

    def forward(self, x, idx_client=0):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        assert idx_client <= self.num_client-1
        client_name = 'classifier_' + str(idx_client+1)
        mapping_net = getattr(self, client_name)
        x, global_r = mapping_net(x)
        return x, global_r


# Feature embedding network
class embedding_disEN_net_glob(nn.Module):
    def __init__(self, num_ids_client, feat_dim=2048):
        super(embedding_disEN_net_glob, self).__init__()
        model_backbone = models.resnet50(pretrained=True)
        model_backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_backbone
        self.num_client = len(num_ids_client)

        self.classifier = []
        for i in range(len(num_ids_client)):
            name = 'classifier_'+str(i+1)
            setattr(self, name, MLP_disentangle_glob(feat_dim, num_ids_client[i]))

    def forward(self, x, idx_client=0):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))

        assert idx_client <= self.num_client-1
        client_name = 'classifier_' + str(idx_client+1)
        mapping_net = getattr(self, client_name)
        x = mapping_net(x)
        return x

# Feature embedding network for testing
class embedding_net_test(nn.Module):
    def __init__(self, model):
        super(embedding_net_test, self).__init__()
        self.model = model.model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1)) # embedding feature representation
        return x

    
