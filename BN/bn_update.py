import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm

import torch.nn.functional as F

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    #model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            #momentum = b / (n + b)
            for module in momenta.keys():
                #module.momentum = momentum
                module.momentum = 0.5
            
            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    print('batch normalization update finish ')
    return n

def bn_update_late(loader, model,n, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """

    if not check_bn(model):
        return
    model.train()
    momenta = {}
    #model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = n
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            #momentum = b / (n + b)
            for module in momenta.keys():
                #module.momentum = momentum
                module.momentum = 0.5
            
            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    print('batch normalization update finish ')