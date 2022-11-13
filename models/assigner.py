"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import rich
import numpy as np
import math
import time
from scipy.optimize import linear_sum_assignment

from utils.torch_utils import de_parallel
from utils.metrics import bbox_iou, pairwise_bbox_iou, pairwise_kpts_oks
from utils.general import LOGGER, colorstr, xywh2xyxy, crop_mask



class Assigner:
    def __init__(self)
        pass


    def forward(self, preds, targets, masks=None, epoch=0, epochs=0):
        pass



