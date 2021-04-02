from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import numpy as np
from scipy.spatial.distance import cdist, pdist

# from config import opt
# # global variables
#
# args = opt

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.SoftMarginLoss()#nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  #if not max, size=(64,4)
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  #if not min, size=(64,60)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an-dist_ap, y)
        return loss


class TripletLoss_mutual_sr(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, args, margin=0.3):
        super(TripletLoss_mutual_sr, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.SoftMarginLoss()#nn.MarginRankingLoss(margin=0.3)
       # global args
        self.args = args

    def forward(self, inputs, sr_p, sr_n, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # if not max, size=(64,4)
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # if not min, size=(64,60)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute pairwise sr_positive distance
        dist_sr_p = torch.pow(sr_p, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_sr_p = dist_sr_p + dist_sr_p.t()
        dist_sr_p.addmm_(1, -2, sr_p, sr_p.t())
        dist_sr_p = dist_sr_p.clamp(min=1e-12).sqrt()  # for numerical stability
        dist_sr_p_ap, dist_sr_p_an = [], []
        for i in range(n):
            dist_sr_p_ap.append(dist_sr_p[i][mask[i]].max().unsqueeze(0))  # if not max, size=(64,4)
            dist_sr_p_an.append(dist_sr_p[i][mask[i] == 0].min().unsqueeze(0))  # if not min, size=(64,60)
        dist_sr_p_ap = torch.cat(dist_sr_p_ap)
        dist_sr_p_an = torch.cat(dist_sr_p_an)

        # Compute pairwise sr_negative distance
        dist_sr_n = torch.pow(sr_n, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_sr_n = dist_sr_n + dist_sr_n.t()
        dist_sr_n.addmm_(1, -2, sr_n, sr_n.t())
        dist_sr_n = dist_sr_n.clamp(min=1e-12).sqrt()  # for numerical stability
        dist_sr_n_ap, dist_sr_n_an = [], []
        for i in range(n):
            dist_sr_n_ap.append(dist_sr_n[i][mask[i]].max().unsqueeze(0))  # if not max, size=(64,4)
            dist_sr_n_an.append(dist_sr_n[i][mask[i] == 0].min().unsqueeze(0))  # if not min, size=(64,60)
        dist_sr_n_ap = torch.cat(dist_sr_n_ap)
        dist_sr_n_an = torch.cat(dist_sr_n_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        args = self.args
        if args.only_P:
            loss = self.ranking_loss(dist_ap - dist_sr_p_ap, y) + self.ranking_loss(dist_sr_n_ap - dist_ap, y)
        elif args.only_N:
            loss = self.ranking_loss(dist_sr_p_an - dist_an, y) + self.ranking_loss(dist_an - dist_sr_n_an, y)
        elif args.only_Sr:
            loss = self.ranking_loss(-dist_sr_p_ap, y) + self.ranking_loss(dist_sr_n_ap, y) + \
               self.ranking_loss(dist_sr_p_an, y) + self.ranking_loss(-dist_sr_n_an, y)
        elif args.only_Sr_type2:
            loss = self.ranking_loss(dist_sr_n_ap-dist_sr_p_ap, y) + \
               self.ranking_loss(dist_sr_p_an-dist_sr_n_an, y)
        else:
            loss = self.ranking_loss(dist_ap-dist_sr_p_ap, y) + self.ranking_loss(dist_sr_n_ap-dist_ap, y) + \
               self.ranking_loss(dist_sr_p_an-dist_an, y) + self.ranking_loss(dist_an-dist_sr_n_an, y)

        return loss


class TripletLoss_for_style(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Xin Jin
    """

    def __init__(self, margin=0.3):
        super(TripletLoss_for_style, self).__init__()
        self.margin = margin
        self.ranking_loss4useless = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss4useful = nn.MarginRankingLoss(margin=0.5)


    def forward(self, inputs_useful, inputs_useless, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs_useful.size(0)

        # Compute pairwise distance for useful inner features
        dist_useful = torch.pow(inputs_useful, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_useful = dist_useful + dist_useful.t()
        dist_useful.addmm_(1, -2, inputs_useful, inputs_useful.t())
        dist_useful = dist_useful.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask_useful = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap_useful, dist_an_useful = [], []
        for i in range(n):
            dist_ap_useful.append(dist_useful[i][mask_useful[i]].max().unsqueeze(0))
            dist_an_useful.append(dist_useful[i][mask_useful[i] == 0].min().unsqueeze(0))
        dist_ap_useful = torch.cat(dist_ap_useful)
        dist_an_useful = torch.cat(dist_an_useful)

        # Compute pairwise distance for useless inner features
        dist_useless = torch.pow(inputs_useless, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_useless = dist_useless + dist_useless.t()
        dist_useless.addmm_(1, -2, inputs_useless, inputs_useless.t())
        dist_useless = dist_useless.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the easiest positive and negative
        mask_useless = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap_useless, dist_an_useless = [], []
        for i in range(n):
            dist_ap_useless.append(dist_useless[i][mask_useless[i]].min().unsqueeze(0))
            dist_an_useless.append(dist_useless[i][mask_useless[i] == 0].max().unsqueeze(0))
        dist_ap_useless = torch.cat(dist_ap_useless)
        dist_an_useless = torch.cat(dist_an_useless)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an_useful)
        loss = self.ranking_loss4useful(dist_an_useful, dist_ap_useful, y) \
               + self.ranking_loss4useless(dist_ap_useless, dist_an_useless, y)
        return loss


class TripletLoss_meta(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    here, Distance is not mse, is similarity score!!!, so min/max should be inverse

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss_meta, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: distance matrix with shape (batch_size, batch_size)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        dist = inputs
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].min().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].max().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_ap, dist_an, y)
        return loss

class TripletLoss_meta_finetune(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    here, Distance is not mse, is similarity score!!!, so min/max should be inverse

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss_meta_finetune, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs_p, inputs_n, targets_p):
        """
        Args:
        - inputs: distance matrix with shape (batch_size, batch_size)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs_p.size(0)

        dist_p = inputs_p
        dist_p = dist_p.clamp(min=1e-12).sqrt()  # for numerical stability
        dist_n = inputs_n
        dist_n = dist_n.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask_p = targets_p.expand(n, n).eq(targets_p.expand(n, n).t())
        dist_ap = []
        for i in range(n):
            dist_ap.append(dist_p[i][mask_p[i]].min().unsqueeze(0))

        mask_n = torch.ones((n,n))
        mask_n = mask_n.expand(n, n).eq(mask_n.expand(n, n).t()) # should be All 0
        dist_an = []
        for i in range(n):
            dist_an.append(dist_n[i][mask_n[i] != 0].max().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_ap, dist_an, y)
        return loss


class DiscriminativeLoss(torch.nn.Module):
    def __init__(self, mining_ratio=0.001):
        super(DiscriminativeLoss, self).__init__()
        self.mining_ratio = mining_ratio
        self.register_buffer('n_pos_pairs', torch.Tensor([0]))
        self.register_buffer('rate_TP', torch.Tensor([0]))
        self.moment = 0.1
        self.initialized = False

    def init_threshold(self, pairwise_agreements):
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = sorted_agreements[-pos]
        self.register_buffer('threshold', torch.Tensor([t]).cuda())
        self.initialized = True

    def forward(self, features, multilabels, labels):
        """
        :param features: shape=(BS, dim)
        :param multilabels: (BS, n_class)
        :param labels: (BS,)
        :return:
        """
        P, N = self._partition_sets(features.detach(), multilabels, labels)
        if P is None:
            pos_exponant = torch.Tensor([1]).cuda()
            num = 0
        else:
            sdist_pos_pairs = []
            for (i, j) in zip(P[0], P[1]):
                sdist_pos_pair = (features[i] - features[j]).pow(2).sum()
                sdist_pos_pairs.append(sdist_pos_pair)
            pos_exponant = torch.exp(- torch.stack(sdist_pos_pairs)).mean()
            #(pos_exponant.data)
            num = -torch.log(pos_exponant)
            #print(num)
        if N is None:
            neg_exponant = torch.Tensor([0.5]).cuda()
        else:
            sdist_neg_pairs = []
            for (i, j) in zip(N[0], N[1]):
                sdist_neg_pair = (features[i] - features[j]).pow(2).sum()
                sdist_neg_pairs.append(sdist_neg_pair)
            neg_exponant = torch.exp(- torch.stack(sdist_neg_pairs)).mean()
        den = torch.log(pos_exponant + neg_exponant)
        loss = num + den
        return loss

    def _partition_sets(self, features, multilabels, labels):
        """
        partition the batch into confident positive, hard negative and others
        :param features: shape=(BS, dim)
        :param multilabels: shape=(BS, n_class)
        :param labels: shape=(BS,)
        :return:
        P: positive pair set. tuple of 2 np.array i and j.
            i contains smaller indices and j larger indices in the batch.
            if P is None, no positive pair found in this batch.
        N: negative pair set. similar to P, but will never be None.
        """
        f_np = features.cpu().numpy()
        ml_np = multilabels.cpu().detach().numpy()
        p_dist = pdist(f_np)
        p_agree = 1 - pdist(ml_np, 'minkowski', p=1) / 2
        sorting_idx = np.argsort(p_dist)
        n_similar = int(len(p_dist) * self.mining_ratio)
        similar_idx = sorting_idx[:n_similar]
        is_positive = p_agree[similar_idx] > self.threshold.item()
        pos_idx = similar_idx[is_positive]
        neg_idx = similar_idx[~is_positive]
        P = dist_idx_to_pair_idx(len(f_np), pos_idx)
        N = dist_idx_to_pair_idx(len(f_np), neg_idx)
        self._update_threshold(p_agree)
        self._update_buffers(P, labels)
        #print(pos_idx.size)
        #print(neg_idx.size)
        return P, N

    def _update_threshold(self, pairwise_agreements):
        pos = int(len(pairwise_agreements) * self.mining_ratio)
        sorted_agreements = np.sort(pairwise_agreements)
        t = torch.Tensor([sorted_agreements[-pos]]).cuda()
        self.threshold = self.threshold * (1 - self.moment) + t * self.moment

    def _update_buffers(self, P, labels):
        if P is None:
            self.n_pos_pairs = 0.9 * self.n_pos_pairs
            return 0
        n_pos_pairs = len(P[0])
        count = 0
        for (i, j) in zip(P[0], P[1]):
            count += labels[i] == labels[j]
        rate_TP = float(count) / n_pos_pairs
        self.n_pos_pairs = 0.9 * self.n_pos_pairs + 0.1 * n_pos_pairs
        self.rate_TP = 0.9 * self.rate_TP + 0.1 * rate_TP

def dist_idx_to_pair_idx(d, i):
    """
    :param d: number of samples
    :param i: np.array
    :return:
    """
    if i.size == 0:
        return None
    b = 1 - 2 * d
    x = np.floor((-b - np.sqrt(b ** 2 - 8 * i)) / 2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    return x, y

from torch.nn import functional as F
class GDS_loss(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_cosine=True):
        super(GDS_loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.K = num_instances
        self.use_cosine = use_cosine

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged

        if not self.use_cosine:
            # L2 distance:
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        else:
            # Cosine distance:
            input1_normed = F.normalize(inputs, p=2, dim=1)
            input2_normed = F.normalize(inputs, p=2, dim=1)
            dist = 1 - torch.mm(input1_normed, input2_normed.t())

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].unsqueeze(0))        #if not max, size=(64,4)
            dist_an.append(dist[i][mask[i] == 0].unsqueeze(0))   #if not min, size=(64,60)
        dist_ap = torch.cat(dist_ap)   #size=(64,4)
        dist_an = torch.cat(dist_an)   #size=(64,60)

        dist_ap = dist_ap.view(-1)
        dist_an = dist_an.view(-1)

        # Compute mean and std of positive_set and negative_set
        dist_ap_mean = dist_ap.mean()
        dist_an_mean = dist_an.mean()
        dist_ap_std = dist_ap.std()
        dist_an_std = dist_an.std()

        return dist_ap_mean, dist_ap_std, dist_an_mean, dist_an_std