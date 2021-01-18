import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class MergeLoss(nn.Module):

    def __init__(self,loss_nr):
        super(MergeLoss, self).__init__()
        self.log_var_list = []

        cuda0 = torch.device('cuda:0')
        self.log_var_0 = torch.zeros([1], dtype=torch.float64, device=cuda0, requires_grad=True)
        self.log_var_1 = torch.zeros([1], dtype=torch.float64, device=cuda0, requires_grad=True)
        self.log_var_2 = torch.zeros([1], dtype=torch.float64, device=cuda0, requires_grad=True)

    def forward(self, loss_list):
        loss=0
        for id,loss_i in enumerate(loss_list):
            log_var = getattr(self, 'log_var_{}'.format(id))
            precision = torch.exp(-log_var)
            loss+=torch.sum(precision*loss_i+log_var,-1)
        return torch.mean(loss)

    def get_data(self):
        print('==>',self.log_var_0,self.log_var_1,self.log_var_2)
        print('exp result ==>',torch.exp(-self.log_var_0),torch.exp(-self.log_var_1),torch.exp(-self.log_var_2))

        return torch.exp(-self.log_var_0),torch.exp(-self.log_var_1),torch.exp(-self.log_var_2)



