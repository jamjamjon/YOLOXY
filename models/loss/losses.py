"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import rich
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

from utils.torch_utils import de_parallel
from utils.metrics import bbox_iou, pairwise_bbox_iou
from utils.general import CONSOLE, LOGGER, colorstr



class VariFL(nn.Module):
    """Varifocal Loss <https://arxiv.org/abs/2008.13367>"""
    def __init__(self, gamma=2.0, alpha=0.75, reduction="mean"): 
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        assert pred.size() == target.size()
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        focal_weight = target * (target > 0.0).float() + self.alpha * (pred_sigmoid - target).abs().pow(self.gamma) * (target <= 0.0).float()
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction=self.reduction) * focal_weight
        return loss




class OKSLoss(nn.Module):
    # Objects Keypoints Similarity Loss

    def __init__(self, sigmas):
        super().__init__()

        self.sigmas = sigmas
        self.BCEkpt = nn.BCEWithLogitsLoss(reduction="none")  # kpt
        LOGGER.info(f"{colorstr('Keypoints Sigmas: ')} {[x for x in sigmas.cpu().numpy()]}")


    def forward(self, pkpt, tkpt, tbox, alpha=1.0, beta=2.0):

        # x, y, conf
        pkpt_x, pkpt_y, pkpt_score = pkpt[:, 0::3], pkpt[:, 1::3], pkpt[:, 2::3]
        tkpt_x, tkpt_y = tkpt[:, 0::2], tkpt[:, 1::2]   

        # loss of kpts conf => visibility
        tkpt_mask = (tkpt[:, 0::2] != 0)     # visibilty flag are used as GT
        lkpt_conf = self.BCEkpt(pkpt_score, tkpt_mask.float()).mean(axis=1)
        
        # loss of kpts => oks
        d = (pkpt_x - tkpt_x) ** 2 + (pkpt_y - tkpt_y) ** 2   # L2 distance
        s = torch.prod(tbox[:, -2:], dim=1, keepdim=True)  # scale(area) of GT bbox 
        kpt_loss_factor = (torch.sum(tkpt_mask != 0) + torch.sum(tkpt_mask == 0)) / torch.sum(tkpt_mask != 0)

        # https://github.com/TexasInstruments/edgeai-yolov5/blob/yolo-pose/utils/loss.py 
        # official: lambda_kpt = 0.1, lambda_kpt_conf = 0.5, lambda_box = 0.05, lambda_cls = 0.5
        lkpt = kpt_loss_factor * ((1 - torch.exp(-d / (s * (4 * self.sigmas ** 2) + 1e-9))) * tkpt_mask).mean(axis=1)

        # deprecated.
        # oks = torch.exp(-d / (s * (4 * self.sigmas) + 1e-9))   
        # lkpt = kpt_loss_factor * ((1 - oks ** 2) * tkpt_mask).mean(axis=1)

        return lkpt * alpha + lkpt_conf * beta


