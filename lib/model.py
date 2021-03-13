# 
# FedReID model: feature embedding network + mapping network
#

import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import torch
from SNR.resnet_SNR import resnet50_SNR

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


    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)


# Feature embedding network 
class embedding_net(nn.Module):
    def __init__(self, num_ids_client, feat_dim=2048):
        super(embedding_net, self).__init__()
        model_backbone = resnet50_SNR(loss={'xent', 'htri'})
        model_backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_backbone
        self.num_client = len(num_ids_client)

        self.classifier = []
        for i in range(len(num_ids_client)):
            name = 'classifier_'+str(i+1)
            setattr(self, name, MLP(feat_dim, num_ids_client[i]))

    def forward(self, x, idx_client=0):
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = self.model.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        if self.model.training:
            features, f4_useful, f3_useful, f2_useful, f1_useful, \
            f4_useless, f3_useless, f2_useless, f1_useless, \
            f4_IN, f3_IN, f2_IN, f1_IN, \
            = self.model(x)
            assert idx_client <= self.num_client-1
            client_name = 'classifier_' + str(idx_client+1)
            mapping_net = getattr(self, client_name)
            outputs = mapping_net(features)
            return outputs,features, f4_useful, f3_useful, f2_useful, f1_useful, \
            f4_useless, f3_useless, f2_useless, f1_useless, \
            f4_IN, f3_IN, f2_IN, f1_IN
        else: # in evaluation mode
            features = self.model(x)
            assert idx_client <= self.num_client - 1
            client_name = 'classifier_' + str(idx_client + 1)
            mapping_net = getattr(self, client_name)
            outputs = mapping_net(features)
            return outputs
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

