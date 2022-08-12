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


class VarifocalLoss(nn.Module):

    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self,
                pred_score,
                gt_score,
                label,
                alpha=0.75,
                gamma=2.0):
        """
        仅适用于当前任务。调用binary_cross_entropy不进行reduction。后乘上weight，再进行sum
        :param pred_score:
        :param gt_score:
        :param label:
        :param alpha:
        :param gamma:
        :return:
        """
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = (F.binary_cross_entropy(pred_score, gt_score, reduction='none') * weight).sum()

        return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
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


