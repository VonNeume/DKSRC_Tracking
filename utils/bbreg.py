import torch
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from utils.sample_geneator import overlap_ratio


class BBRegressor():
    def __init__(self, img_size, alpha=1e4, overlap=[0.6, 1], scale=[1, 2]):
        self.img_size = img_size
        self.alpha = alpha
        self.overlap_range = overlap
        self.scale_range = scale
        self.model = KernelRidge(alpha=self.alpha) # KernelRidge(alpha=self.alpha)

    def train(self, X, bbox, gt):
        X = X.cpu().detach().numpy()
        bbox = np.copy(bbox)
        gt = np.copy(gt)

        if gt.ndim == 1:
            gt = gt[None, :]

        r = overlap_ratio(bbox, gt)
        s = np.prod(bbox[:, 2:], axis=1) / np.prod(gt[0, 2:])
        idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
              (s >= self.scale_range[0]) * (s <= self.scale_range[1])

        X = X[idx]
        bbox = bbox[idx]

        Y = self.get_examples(bbox, gt)

        self.model.fit(X, Y)

    def predict(self, X, bbox):
        X = X.cpu().detach().numpy()
        bbox_ = np.copy(bbox)

        Y = self.model.predict(X)

        bbox_[:, :2] = bbox_[:, :2] + bbox_[:, 2:] / 2
        bbox_[:, :2] = Y[:, :2] * bbox_[:, 2:] + bbox_[:, :2]
        bbox_[:, 2:] = np.exp(Y[:, 2:]) * bbox_[:, 2:]
        bbox_[:, :2] = bbox_[:, :2] - bbox_[:, 2:] / 2

        r = overlap_ratio(bbox, bbox_)
        s = np.prod(bbox[:, 2:], axis=1) / np.prod(bbox_[:, 2:], axis=1)
        idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
              (s >= self.scale_range[0]) * (s <= self.scale_range[1])
        idx = np.logical_not(idx)
        bbox_[idx] = bbox[idx]

        bbox_[:, :2] = np.maximum(bbox_[:, :2], 0)
        bbox_[:, 2:] = np.minimum(bbox_[:, 2:], self.img_size - bbox[:, :2])

        return bbox_

    def get_examples(self, bbox, gt):
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:] / 2
        gt[:, :2] = gt[:, :2] + gt[:, 2:] / 2

        dst_xy = (gt[:, :2] - bbox[:, :2]) / bbox[:, 2:]
        dst_wh = np.log(gt[:, 2:] / bbox[:, 2:])

        Y = np.concatenate((dst_xy, dst_wh), axis=1)
        return Y


class BBRegressorNN():
    def __init__(self, img_size, alpha=1e4, overlap=[0.6, 1], scale=[1, 2]):
        self.img_size = img_size
        self.alpha = alpha
        self.overlap_range = overlap
        self.scale_range = scale
        self.model = BBregNet(512).cuda()
        ckpt = torch.load('checkpoints/bk_checkpoint.pth.tar')
        self.model.load_state_dict(ckpt['state_dict'])

    def predict(self, X, bbox):
        X = X
        bbox_ = np.copy(bbox)

        Y = self.model(X)
        Y = Y.detach().cpu().numpy()

        bbox_[:, :2] = bbox_[:, :2] + bbox_[:, 2:] / 2
        bbox_[:, :2] = Y[:, :2] * bbox_[:, 2:] + bbox_[:, :2]
        bbox_[:, 2:] = np.exp(Y[:, 2:]) * bbox_[:, 2:]
        bbox_[:, :2] = bbox_[:, :2] - bbox_[:, 2:] / 2

        r = overlap_ratio(bbox, bbox_)
        s = np.prod(bbox[:, 2:], axis=1) / np.prod(bbox_[:, 2:], axis=1)
        idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
              (s >= self.scale_range[0]) * (s <= self.scale_range[1])
        idx = np.logical_not(idx)
        bbox_[idx] = bbox[idx]

        bbox_[:, :2] = np.maximum(bbox_[:, :2], 0)
        bbox_[:, 2:] = np.minimum(bbox_[:, 2:], self.img_size - bbox[:, :2])

        return bbox_

    def get_examples(self, bbox, gt):
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:] / 2
        gt[:, :2] = gt[:, :2] + gt[:, 2:] / 2

        dst_xy = (gt[:, :2] - bbox[:, :2]) / bbox[:, 2:]
        dst_wh = np.log(gt[:, 2:] / bbox[:, 2:])

        Y = np.concatenate((dst_xy, dst_wh), axis=1)
        return Y

class BBregNet(nn.Module):
    def __init__(self, in_dim=64):
        super(BBregNet, self).__init__()

        self.fc1 = nn.Linear(in_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.boxreg = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        boxreg = self.boxreg(x)

        return boxreg