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
from utils.general import CONSOLE, LOGGER, colorstr, xywh2xyxy, crop_mask



def distribution_focal_loss(pred, label):
    """Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


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
        pkpt_x, pkpt_y, pkpt_conf = pkpt[:, 0::3], pkpt[:, 1::3], pkpt[:, 2::3]
        tkpt_x, tkpt_y = tkpt[:, 0::2], tkpt[:, 1::2]   

        # loss of kpts conf => visibility
        tkpt_mask = (tkpt[:, 0::2] != 0)     # visibilty flag are used as GT
        lkpt_conf = self.BCEkpt(pkpt_conf, tkpt_mask.float()).mean(axis=1)
        
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




class ComputeLoss:
    # SimOTA
    def __init__(self, model):
        self.device = next(model.parameters()).device  # get model device
        self.hyp = model.hyp  # hyperparameters
        self.head = de_parallel(model).model[-1]  # Detect() module
        self.ng = 0   # number of grid in every scale: 80x80 + 40x40 + 20x20

        # head attrs
        for x in ('nl', 'stride', 'na', 'nc', 'nk', 'no', 'no_det', 'no_kpt'):
            setattr(self, x, getattr(self.head, x))
        
        # Define criteria
        # self.LossFn_CLS = VariFL(gamma=2.0, alpha=0.75, reduction="none")   # VFL 
        self.LossFn_CLS = nn.BCEWithLogitsLoss(reduction="none")   # reduction="mean" default, pos_weights=None
        self.LossFn_OBJ = nn.BCEWithLogitsLoss(reduction="none")   # TODO: add pos_weights=None
        self.L1_BOX = nn.L1Loss(reduction="none")
        if self.nk > 0:
            kpts_weights = self.hyp.get('kpt_weights', None)
            if kpts_weights is None:
                LOGGER.info(f"{colorstr('magenta', 'b', 'Attention: ')}Weights of Each Keypoint Is Not Set. Do It At => data/hyps/x.yaml")
                kpts_weights = (torch.tensor([.1] * self.nk)).to(self.device)
            else:
                if len(kpts_weights) == self.nk:
                    kpts_weights = (torch.tensor(kpts_weights)).to(self.device)
                else:
                    kpts_weights = (torch.tensor([.1] * self.nk)).to(self.device)
                # assert len(kpts_weights) == self.nk, f"Number of kpts weights {len(kpts_weights)} not matched with self.nk {self.nk}!"
            self.LossFn_KPT = OKSLoss(kpts_weights)
            self.kpts_sigmas = kpts_weights


    def __call__(self, p, targets, masks=None, epoch=0, epochs=0):
        # p: {(bs, 1, 80, 80, no), (bs, 1, 40, 40, no), ...}
        # targets: { num_object, 6 + no_kpt(idx, cls, xywh, kpts(optional)) }   

        # loss item init
        lcls = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device) 
        lbox_l1 = torch.zeros(1, device=self.device)
        lkpt = torch.zeros(1, device=self.device)
        lseg = torch.zeros(1, device=self.device)

        # build targets
        (   pbox, pbox0, pobj, pcls, pkpt,
            tcls, tbox, tbox_l1, tobj, tkpt,
            finalists_masks, num_finalists
        ) = self.build_targets(p, targets)

        # compute loss
        lbox += (1.0 - bbox_iou(pbox.view(-1, 4)[finalists_masks], tbox, SIoU=True).squeeze()).sum() / num_finalists  # iou(prediction, target)
        lbox_l1 += (self.L1_BOX(pbox0.view(-1, 4)[finalists_masks], tbox_l1)).sum() / num_finalists
        lobj += (self.LossFn_OBJ(pobj.view(-1, 1), tobj)).sum() / num_finalists
        lcls += (self.LossFn_CLS(pcls.view(-1, self.nc)[finalists_masks], tcls)).sum() / num_finalists
        if self.nk > 0 and pkpt is not None and tkpt is not None:   # kpt loss
            # -------------------------
            #   OKS Loss for kpts  
            #   TODO: Wingloss or SmoothL1 loss, ... for other kpts task 
            # -------------------------
            lkpt += self.LossFn_KPT(pkpt.view(-1, self.no_kpt)[finalists_masks], tkpt, tbox).sum() / num_finalists


        # loss weighted
        lbox *= self.hyp['box']    # self.hyp.get('box', 5.0)    
        lbox_l1 *= self.hyp['box_l1']    # self.hyp.get('box_l1', 1.0)    
        lbox += lbox_l1
        lcls *= self.hyp['cls']    # self.hyp.get('cls', 1.0)
        lobj *= self.hyp['obj']    # self.hyp.get('obj', 1.0)
        lkpt *= self.hyp['kpt']    # self.hyp.get('kpt', 5.5)    
        lseg *= 1   # not now 

        return lbox + lobj + lcls + lkpt + lseg, torch.cat((lbox, lobj, lcls, lkpt, lseg)).detach()  


    # build predictions
    def build_preds(self, p):
        
        xy_shifts, expanded_strides, preds_new, preds_scale = [], [], [], []

        for k, pred in enumerate(p):
            # ------------------------------------------------------------------
            # decode pred: [bs, 1, 80, 80, no] => [bs, 8400, no]
            # ------------------------------------------------------------------
            bs, _, h, w, _ = pred.shape   # [bs, na, 80, 80, no]
            grid = self.head.grid[k].to(self.device)    # [80， 40， 20] 

            # grid init at the 1st time
            if grid.shape[2:4] != pred.shape[2:4]:
                grid = self.head._make_grid(w, h).to(self.device)
                self.head.grid[k] = grid    # [1, 1, 80, 80, 2]

            pred = pred.reshape(bs, self.na * h * w, -1)    # （bs, 80x80, -1）
            pred_scale = pred.clone()   # clone

            # de-scale to img size
            xy_shift = grid.view(1, -1, 2)  # [1, 8400, 2])  grid_xy
            pred[..., :2] = (pred[..., :2] + xy_shift) * self.stride[k]     # xy
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.stride[k]     # wh

            # kpt
            if self.nk > 0:
                kpt_conf_grids = torch.zeros_like(xy_shift)[..., 0:1]   #  [1, 8400, 1]
                kpt_grids = torch.cat((xy_shift, kpt_conf_grids), dim=2).repeat(1, 1, self.nk)  #  [1, 8400, 3 * nk]  
                pred[..., -3 * self.nk:] = (pred[..., -3 * self.nk:] + kpt_grids) * self.stride[k]
            # ------------------------------------------------------------------

            # stride between grid 
            expanded_stride = torch.full((1, xy_shift.shape[1], 1), self.stride[k], device=self.device)     #[1, 6400, 1]

            # append to list
            xy_shifts.append(xy_shift)
            expanded_strides.append(expanded_stride)
            preds_new.append(pred)              # [[16, 6400, 85], [16, 1600, 85], [16, 400, 85]]
            preds_scale.append(pred_scale)      # [[16, 6400, 85], [16, 1600, 85], [16, 400, 85]]

        # concat
        xy_shifts = torch.cat(xy_shifts, 1)                 # [1, n_anchors_all(8400), 2]
        expanded_strides = torch.cat(expanded_strides, 1)   # [1, n_anchors_all(8400), 1]
        preds_scale = torch.cat(preds_scale, 1)             # [16, 8400, 85]
        p = torch.cat(preds_new, 1)                     # [16, 8400, 85]
        self.ng = p.shape[1]      # 80x80 + 40x40 + 20x20

        pbox = p[:, :, :4]                  # at input size. [batch, n_anchors_all, 4]
        pbox0 = preds_scale[:, :, :4]       # at scales, for l1 loss compute. [batch, n_anchors_all, 4]
        pobj = p[:, :, 4].unsqueeze(-1)     # [batch, n_anchors_all, 1]
        pcls = p[:, :, 5: self.no_det]      # [batch, n_anchors_all, n_cls]
        
        # kpt
        if self.nk > 0:
            pkpt = p[:, :, self.no_det:]  # [batch, n_anchors_all, nk*3]
        else:
            pkpt = None

        return p, pbox, pbox0, pobj, pcls, pkpt, xy_shifts, expanded_strides


    # build targets
    def build_targets(self, p, targets):
        # p: {(bs, 1, 80, 80, 85), ...}
        input_h, input_w = self.stride[0] * p[0].shape[2], self.stride[0] * p[0].shape[3] # 640, 640

        # build predictions
        (   p,                          # [bs, 1, 80, 80, no] => [bs, 8400, no]
            pbox,                       # [batch, n_anchors_all, 4]
            pbox0,                      # [batch, n_anchors_all, 4]
            pobj,                       # [batch, n_anchors_all, 1]
            pcls,                       # [batch, n_anchors_all, n_cls]
            pkpt,                       # [batch, n_anchors_all, nk*3]
            self.xy_shifts,             # [1, n_anchors_all(8400), 2]
            self.expanded_strides,      # [1, n_anchors_all(8400), 1] 
        ) = self.build_preds(p)


        # build targets
        targets_list = np.zeros((p.shape[0], 1, 5 + self.nk * 2)).tolist()   # batch size
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        empty_list = [[-1] + [0] * (self.nk * 2 + 4)]  # cls, xy * self.nk 
        targets = torch.from_numpy(np.array(list(map(lambda l:l + empty_list * (max_len - len(l)), targets_list)))[:,1:,:]).to(self.device)
        nts = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects list per batch [13, 4, 2, ...]

        # targets cls, box, ...
        tcls, tbox, tbox_l1, tobj, finalists_masks, num_finalists = [], [], [], [], [], 0 
        if self.nk > 0:
            tkpt = []
        else:
            tkpt = None
        
        # batch images loop
        for idx in range(p.shape[0]):   # batch size
            nt = int(nts[idx])  # num of targets in current image

            if nt == 0:     # num targets=0  =>  neg sample image
                tcls_ = p.new_zeros((0, self.nc))
                tbox_ = p.new_zeros((0, 4))
                tbox_l1_ = p.new_zeros((0, 4))
                tobj_ = p.new_zeros((self.ng, 1))
                finalists_mask = p.new_zeros(self.ng).bool()
                if self.nk > 0:
                    tkpt_ = p.new_zeros((0, self.nk * 2))  # kpt

            else:   
                imgsz = torch.Tensor([[input_w, input_h, input_w, input_h]]).type_as(targets)  # [[640, 640, 640, 640]]
                t_bboxes = targets[idx, :nt, 1:5].mul_(imgsz)    # gt bbox, de-scaled 
                t_classes = targets[idx, :nt, 0]   # gt cls [ 0., 40., 23., 23.]
                p_bboxes = pbox[idx]        # pred bbox
                p_classes = pcls[idx]       # pred cls
                p_objs = pobj[idx]          # pred obj
 
                if self.nk > 0:
                    p_kpts = pkpt[idx]          # pred kpts

                    imgsz_kpt = torch.Tensor([[input_w, input_h] * self.nk]).type_as(targets)  # de-scale to origin image size [[640, 640, 640, 640]]
                    t_kpts = targets[idx, :nt, -2 * self.nk:].mul_(imgsz_kpt)  # t_kpts
                else:
                    t_kpts = None
                    p_kpts = None          # pred kpts

                # do label assignment: SimOTA 
                (
                    finalists_mask,
                    num_anchor_assigned,   
                    tcls_, 
                    tobj_, 
                    tbox_, 
                    tbox_l1_,
                    tkpt_
                 ) = self.get_assignments(p_bboxes, p_classes, p_objs, t_bboxes, t_classes, t_kpts, p_kpts)
                
                # num of assigned anchors in one batch
                num_finalists += num_anchor_assigned    

            # append to list
            tcls.append(tcls_)
            tbox.append(tbox_)
            tobj.append(tobj_)
            tbox_l1.append(tbox_l1_)
            finalists_masks.append(finalists_mask)

            # kpt
            if self.nk > 0 and tkpt_ is not None:
                tkpt.append(tkpt_)

        # concat
        tcls = torch.cat(tcls, 0)
        tbox = torch.cat(tbox, 0)
        tobj = torch.cat(tobj, 0)
        tbox_l1 = torch.cat(tbox_l1, 0)
        finalists_masks = torch.cat(finalists_masks, 0)
        num_finalists = max(num_finalists, 1)

        # kpt
        if self.nk > 0:
            tkpt = torch.cat(tkpt, 0)

        return pbox, pbox0, pobj, pcls, pkpt, tcls, tbox, tbox_l1, tobj, tkpt, finalists_masks, num_finalists 


    # SimOTA
    @torch.no_grad()
    def get_assignments(self, p_bboxes, p_classes, p_objs, t_bboxes, t_classes, t_kpts=None, p_kpts=None):

        num_objects = t_bboxes.shape[0]   # number of gt object per image
        candidates_mask, is_in_boxes_and_center = self.get_candidates(t_bboxes)

        # 2. pick preds in fixed center region, and get bbox, cls, obj
        p_bboxes = p_bboxes[candidates_mask]
        cls_preds_ = p_classes[candidates_mask]
        obj_preds_ = p_objs[candidates_mask]
        num_in_boxes_anchor = p_bboxes.shape[0]

        # not neccessary
        # if p_kpts is not None:
        #     p_kpts = p_kpts[candidates_mask]   # [num, 3*nk]

        # 3. iou loss => iou(gts, preds), for calculate dynamic_k
        pair_wise_ious = pairwise_bbox_iou(t_bboxes, p_bboxes, box_format='xywh')
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # 4. cls loss = cls * obj
        gt_cls_per_image = (F.one_hot(t_classes.to(torch.int64), self.nc)
                            .float()
                            .unsqueeze(1)
                            .repeat(1, num_in_boxes_anchor, 1))   # gt classes to one hot

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().sigmoid_().unsqueeze(0).repeat(num_objects, 1, 1) 
                          * obj_preds_.float().sigmoid_().unsqueeze(0).repeat(num_objects, 1, 1))

            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_, obj_preds_

        # kpt loss matrix (OKS)
        # if t_kpts is not None and p_kpts is not None:
        #     pair_wise_kpts = pairwise_kpts_oks(self.kpts_sigmas, p_kpts, t_kpts, t_bboxes, alpha=1.0, beta=2.0)
        #     pairwise_kpts_oks_loss = -torch.log(pair_wise_kpts + 1e-8)
        #     pairwise_kpts_oks_loss = pairwise_kpts_oks_loss.to(self.device)

        # 5. cost ==> cls * 1.0 + iou * 3.0 + neg * 1e+5
        cost = (1.0 * pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 1e6 * (~is_in_boxes_and_center))  
        # cost = (pair_wise_cls_loss ** 1.0 + pair_wise_ious_loss ** 2.0 + 1e6 + (~is_in_boxes_and_center.float()))  

        # print(f'pair_wise_cls_loss: {pair_wise_cls_loss.mean() * 2.0, pair_wise_cls_loss.min(), pair_wise_cls_loss.max()}')
        # print(f'pair_wise_ious_loss: {pair_wise_ious_loss.mean() ** 2.0, pair_wise_ious_loss.min(), pair_wise_ious_loss.max()}')
        # print(f'is_in_boxes_and_center: {(is_in_boxes_and_center.float() + 1e4).max()}')


        del pair_wise_ious_loss, pair_wise_cls_loss

        # if t_kpts is not None and p_kpts is not None:
        #     cost += pairwise_kpts_oks_loss * 5.0        
        #     del pairwise_kpts_oks_loss

        # 6. assign different k positive samples for every gt.
        (   
            num_anchor_assigned,
            pred_ious_this_matching,
            matched_gt_inds,
            finalists_mask     
        ) = self.dynamic_k_matching(cost, pair_wise_ious, t_classes, candidates_mask)
        del cost, pair_wise_ious

        # 7. empty cuda cache
        torch.cuda.empty_cache() 

        # 8. has anchor point assigned
        if num_anchor_assigned > 0:
            # tcls, tbox, tobj
            tcls_ = t_classes[matched_gt_inds]
            tcls_ = F.one_hot(tcls_.to(torch.int64), self.nc) * pred_ious_this_matching.unsqueeze(-1)
            tobj_ = finalists_mask.unsqueeze(-1)  * 1.0
            tbox_ = t_bboxes[matched_gt_inds]

            # tbox_l1, do scale
            tbox_l1_ = p_bboxes.new_zeros((num_anchor_assigned, 4))
            stride_ = self.expanded_strides[0][finalists_mask]
            grid_ = self.xy_shifts[0][finalists_mask]
            tbox_l1_[:, :2] = t_bboxes[matched_gt_inds][:, :2] / stride_ - grid_
            tbox_l1_[:, 2:4] = torch.log(t_bboxes[matched_gt_inds][:, 2:4] / stride_ + 1e-8)

            # kpt
            if t_kpts is not None:
                tkpt_ = t_kpts[matched_gt_inds]
            else:
                tkpt_ = None

        return finalists_mask, num_anchor_assigned, tcls_, tobj_, tbox_, tbox_l1_, tkpt_



    # get candidates: a fixed center region
    def get_candidates(self, t_bboxes):
        num_object = t_bboxes.shape[0]  # number of objects
        grids_stride = self.expanded_strides[0]  # grid stride
        grids_xy = self.xy_shifts[0] * grids_stride  # grid coords in each scale: [0, 1, 2, ...] => [0, 8, 16, ...]
        
        # grid center xy coords
        grids_center = (grids_xy + 0.5 * grids_stride).unsqueeze(0).repeat(num_object, 1, 1) # [n_anchor, 2] -> [n_gt, n_anchor, 2]

        # gt top-left & bottom-right coords
        t_bboxes_tl = (t_bboxes[:, 0:2] - 0.5 * t_bboxes[:, 2:4]).unsqueeze(1).repeat(1, self.ng, 1)
        t_bboxes_br = (t_bboxes[:, 0:2] + 0.5 * t_bboxes[:, 2:4]).unsqueeze(1).repeat(1, self.ng, 1) # [n_gt, 2] -> [n_gt, n_anchor, 2]
        
        # check if grid's center is in gt's box
        bbox_deltas = torch.cat([grids_center - t_bboxes_tl, t_bboxes_br - grids_center], 2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # in fixed region
        # TODO: x percentage of width or height, rather than fixed value: 2.5 
        center_radius = 2.5   # strides=[8, 16, 32] ; 2.5 * strides=[20, 40, 80] = grid_size! that's why 2.5 is better
        t_bboxes_tl = (t_bboxes[:, 0:2]).unsqueeze(1).repeat(1, self.ng, 1) - center_radius * grids_stride.unsqueeze(0)
        t_bboxes_br = (t_bboxes[:, 0:2]).unsqueeze(1).repeat(1, self.ng, 1) + center_radius * grids_stride.unsqueeze(0)

        # check if grid's center is in fixed region
        center_deltas = torch.cat([grids_center - t_bboxes_tl, t_bboxes_br - grids_center], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in gt boxes or in centers region
        is_in_boxes_or_center = is_in_boxes_all | is_in_centers_all

        # in gt boxes and in centers region
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_or_center] & is_in_centers[:, is_in_boxes_or_center]
        
        return is_in_boxes_or_center, is_in_boxes_and_center


    # assign different k positive samples for every gt 
    def dynamic_k_matching(self, cost, pair_wise_ious, t_classes, candidates_mask):

        torch.cuda.empty_cache()    # empty cuda cache at start
        ious_in_boxes_matrix = pair_wise_ious   # iou matrix 

        # get top-k(10) iou anchor point
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        
        # calc dynamic-K for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()

        # create a matching matrix for gt and anchor point(row: gt, col: anchor)
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        
        # assign num of K anchor points for each gt, based on cost matrix
        for gt_idx in range(t_classes.shape[0]):   # number of gt objects
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        # deal with conflict: filter out the anchor point has been assigned to many gts
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        #----------------------------------------------------------------------------------------------------------------
        # TODO: CPU GPU data copy
        # simota enhancement(many2one): assign 1 anchor for gts those have no assigned anchor caused by solving conflict  
        #----------------------------------------------------------------------------------------------------------------
        # if (matching_matrix.sum(1) == 0).sum() > 0:    # some gt(row) has no assigned anchors
        #     cost_non_assigned = cost[matching_matrix.sum(1) == 0, :]
        #     cols_assigned = matching_matrix.sum(0) > 0
        #     cost_non_assigned[:, matching_matrix.sum(0) > 0] = 1E10  

        #     # do linear sum assignment 
        #     cost_non_assigned_cpu = cost_non_assigned.cpu().numpy()
        #     i, j = linear_sum_assignment(cost_non_assigned_cpu)

        #     # create matching matrix non assigned
        #     matching_matrix_non_assigned = torch.zeros_like(cost_non_assigned, dtype=torch.uint8)
        #     matching_matrix_non_assigned[i, j] = 1

        #     # update matching_matrix
        #     matching_matrix[matching_matrix.sum(1) == 0, :] = matching_matrix_non_assigned
        

        #----------------------------------------------------------------------------------------------------------------
        # check again if matching matrix still has conflicts
        # TODO: if still has conflicts, re-assign
        #----------------------------------------------------------------------------------------------------------------
        # assert (matching_matrix.sum(1) == 0).sum() == 0, '>>> Not all GTs have been assigned!'
        # assert (matching_matrix.sum(0) > 1).sum() == 0, '>>> Matching matrix still has conflicts!!!!!!!'
        # deal with conflict: filter out the anchor point has been assigned to many gts
        # anchor_matching_gt = matching_matrix.sum(0)
        # if (anchor_matching_gt > 1).sum() > 0:
        #     print('Matching matrix still has conflicts!!!!!!!')
        #     print(matching_matrix.sum(0))
        #     print(matching_matrix.sum(0) > 1)
        #     print((matching_matrix.sum(0) > 1).sum())

        #     print(f'===> conflict cols:\n {cost[:, anchor_matching_gt > 1]}')
        #----------------------------------------------------------------------------------------------------------------
        

        # get the number of anchor points which have been assigned to all gts
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_anchor_assigned = fg_mask_inboxes.sum().item()

        # update candidates_mask
        candidates_mask[candidates_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        # matching_matrix * iou matrix
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        # finalists_mask
        finalists_mask = candidates_mask

        return num_anchor_assigned, pred_ious_this_matching, matched_gt_inds, finalists_mask





# class ComputeLoss4Segment:
#     # compute loss for instance segmentation
#     def __init__(self, model):
#         self.device = next(model.parameters()).device  # get model device
#         self.hyp = model.hyp  # hyperparameters
#         self.Detect = de_parallel(model).model[-1]  # Detect() module
#         self.ng = 0   # number of grid in every scale: 80x80 + 40x40 + 20x20

#         # head attrs
#         for x in ('nl', 'na', 'nc', 'stride','nm', 'no_det', 'no_mask', 'no'):
#             setattr(self, x, getattr(self.Detect, x))
        
#         # Define criteria
#         # self.LossFn_CLS = VariFL(gamma=2.0, alpha=0.75, reduction="none")   # Vari Focal Loss 
#         self.LossFn_CLS = nn.BCEWithLogitsLoss(reduction="none")   # reduction="mean" default, pos_weights=None
#         self.LossFn_OBJ = nn.BCEWithLogitsLoss(reduction="none")   # TODO: add pos_weights=None
#         self.L1_BOX = nn.L1Loss(reduction="none")


#     def __call__(self, p, targets, masks=None, overlap=True):
#         # p: {(bs, 1, 80, 80, no), (bs, 1, 40, 40, no), ...}
#         # targets: { num_object, 6 + no_kpt(idx, cls, xywh, kpts(optional)) } [n, cls, box]

#         # loss item init
#         lcls = torch.zeros(1, device=self.device)
#         lobj = torch.zeros(1, device=self.device)
#         lbox = torch.zeros(1, device=self.device) 
#         lbox_l1 = torch.zeros(1, device=self.device)
#         lkpt = torch.zeros(1, device=self.device)
#         lseg = torch.zeros(1, device=self.device)

#         # for segment
#         # p_protos  --->  torch.Size([bs, 32, 160, 160])
#         preds, p_protos = p

#         # 
#         self.input_h, self.input_w = self.stride[0] * preds[0].shape[2], self.stride[0] * preds[0].shape[3] # 640, 640

#         # build targets
#         (   pbox, pbox0, pobj, pcls, pseg, 
#             tcls, tbox, tbox_l1, tobj, 
#             finalists_masks, num_finalists, finalists_masks_per_batch, tbox_per_batch
#         ) = self.build_targets(preds, targets)

#         # print(f'tbox_per_batch --->> {tbox_per_batch[0].shape}')  # [2,4]
#         # print(f'tbox_per_batch len --->> {len(tbox_per_batch)}')  

#         # print(f'finalists_masks_per_batch => {len(finalists_masks_per_batch)}')
#         # print(f'finalists_masks_per_batch => {finalists_masks_per_batch[0].shape}')
#         # print(f'finalists_masks_per_batch => {finalists_masks_per_batch[1].shape}')
#         # print(f'finalists_masks_per_batch => {finalists_masks_per_batch[2].shape}')
#         # print(f'finalists_masks => {finalists_masks.shape}')

#         # compute loss
#         lbox += (1.0 - bbox_iou(pbox.view(-1, 4)[finalists_masks], tbox, SIoU=True).squeeze()).sum() / num_finalists  # iou(prediction, target)
#         lbox_l1 += (self.L1_BOX(pbox0.view(-1, 4)[finalists_masks], tbox_l1)).sum() / num_finalists
#         lobj += (self.LossFn_OBJ(pobj.view(-1, 1), tobj)).sum() / num_finalists
#         lcls += (self.LossFn_CLS(pcls.view(-1, self.nc)[finalists_masks], tcls)).sum() / num_finalists
        
#         # segment loss
#         if tuple(masks.shape[-2:]) != tuple(p_protos.shape[-2:]):  # downsample GT masks to P3 size (bs, 640, 640) -> (bs, 160, 160)
#             masks = F.interpolate(masks[None], tuple(p_protos.shape[-2:]), mode='nearest')[0]
            
#         # calc seg loss
#         for b in range(p_protos.shape[0]): 
#             # gt 
#             # if masks.max() > 1.0:  # mean that GT masks are overlaped
#             if overlap:
#                 # print(f'GT mask shape : {masks.shape}')
#                 # print(f'----> overlaped')
#                 # print(f'overlaped masks ---> {masks.shape}')

#                 if masks.max() == 0.0:   # negtive sample
#                     continue

#                 gt_masks = masks[[b]]  # (bs, 160, 160)  each mini-batch mask
#                 # print(f'overlaped gtmask111 ---> {gt_masks.shape}')   # 1,160,160

#                 nl = torch.sum(finalists_masks_per_batch[b]).item()   # mini-batch targets num
#                 index = torch.arange(nl).reshape(nl, 1, 1) + 1     # (nl, 1, 1)
#                 gt_masks = gt_masks.repeat(nl, 1, 1)    # repeate to (n, 160, 160)
#                 gt_masks = torch.where(gt_masks == index.to(self.device), 1.0, 0.0)
#                 # print(f'overlaped gtmask222 ---> {gt_masks.shape}')

#             else:
                
#                 # ValueError: Target size (torch.Size([1, 160, 160])) must be the same as 
#                 # input size (torch.Size([2, 160, 160]))
#                 if masks.max() == 0.0:   # negtive sample
#                     continue 
#                 # print(f'not overlapedmasks ---> {masks.shape}')
#                 # print(f'not overlaped! max ---> {masks.max()}')
#                 # idx = targets[:, 0] == i

#                 gt_masks = masks[[b]]  # TODO: not overlaped


#                 # print(f'not overlaped gtmask111 ---> {gt_masks.shape}')


#             # print(f'processed GT mask shape : {gt_masks.shape}')

#             # preds 
#             mask_coef = pseg[b][finalists_masks_per_batch[b]] #  mini-batch finalist mask coefficients (N, 32)
#             # mask_coef = pseg.view(-1, self.no_mask)[finalists_masks] #  mini-batch finalist mask coefficients (N, 32)

#             # print(f'pseg[b]  ===> {pseg[b].shape}')  # [8400, 32]
#             # print(f'finalists_masks_per_batch[b]  ===> {finalists_masks_per_batch[b].shape}')  # [8400, 32]
#             # print(f'pseg  ===> {pseg.shape}')  # [bs, 8400, 32]
#             # print(f'mask_coef  ===> {mask_coef.shape}')   # [113, 32]  
#             # print(f'pseg[b].view(-1, self.no_mask)[finalists_masks]  ===> {pseg[b].view(-1, self.no_mask)[finalists_masks_per_batch].shape}')   # [113, 32]  

#             # pred mask: mask_coefficient @ proto_pred  => (n, 32) @ (32, 160, 160) -> (n, 160, 160)
#             p_mask = (mask_coef @ p_protos[b].view(self.nm, -1)).view(-1, *(p_protos.shape[-2:]))
#             # print(f'pred mask ===+> {p_mask.shape}')

#             # calc loss 
#             # pred_mask: ([n, 160, 160])
#             # # gt_mask: ([n, 160, 160])
#             # print(f'pred mask ---> {p_mask.shape}')
#             # print(f'gt mask ---> {gt_masks.shape}')

#             lseg_ = F.binary_cross_entropy_with_logits(p_mask, gt_masks, reduction="none")  #  ([n, 160, 160]) 

#             # TODO: 
#             # 1. gt mask xyxy &  box area (GT) normalized 
#             # print(f'tbox_per_batch0 ---> {tbox_per_batch[0].shape}')
#             # print(f'tbox_per_batch1 ---> {tbox_per_batch[1].shape}')
#             # print(f'tbox ---> {tbox.shape}')

#             tboxn = tbox_per_batch[b].mul(torch.Tensor([[1 / self.input_w, 1 / self.input_h] * 2]).type_as(tbox))    # xywh normalized
#             marean = torch.prod(tboxn[:, -2:], dim=1, keepdim=True)  # mask bbox area normalized
#             mxyxyn = xywh2xyxy(tboxn.mul_(torch.Tensor([p_protos.shape[-2: ] * 2]).type_as(tbox)))   # scale to proto size; xywh -> xyxy
#             # print(f'mxyxyn ---> {mxyxyn.shape}')


#             # 3. crop mask 
#             lseg += (crop_mask(lseg_, mxyxyn).mean(dim=(1, 2)) / marean).mean()
#             # print(f'lseg_: {(crop_mask(lseg_, mxyxyn).mean(dim=(1, 2)) / marean).mean()}')
#             # print(f'lseg: {lseg}')


#         # loss weighted
#         lbox *= self.hyp['box']    # self.hyp.get('box', 5.0)    
#         lbox_l1 *= self.hyp['box_l1']    # self.hyp.get('box_l1', 1.0)    
#         lbox += lbox_l1
#         lcls *= self.hyp['cls']    # self.hyp.get('cls', 1.0)
#         lobj *= self.hyp['obj']    # self.hyp.get('obj', 1.0)
#         lkpt *= self.hyp['kpt']    # self.hyp.get('kpt', 5.5)  
#         # lseg *= 1   # TODO : self.hyp['box'] / p_protos.shape[0]
#         lseg *= self.hyp['box'] / p_protos.shape[0]  # TODO : self.hyp['box'] / p_protos.shape[0]


#         return lbox + lobj + lcls + lkpt + lseg, torch.cat((lbox, lobj, lcls, lkpt, lseg)).detach()  


#     # build predictions 
#     def build_preds(self, p):
        
#         xy_shifts, expanded_strides, preds_new, preds_scale = [], [], [], []

#         for k, pred in enumerate(p):
#             # ------------------------------------------------------------------
#             # decode pred: [bs, 1, 80, 80, no(5 + nc80 + nm32) = 117] => [bs, 8400, no]
#             # ------------------------------------------------------------------

#             bs, _, h, w, _ = pred.shape   # [bs, na, 80, 80, no]
#             grid = self.Detect.grid[k].to(self.device)    # [80， 40， 20] 

#             # grid init at the 1st time
#             if grid.shape[2:4] != pred.shape[2:4]:
#                 grid = self.Detect._make_grid(w, h).to(self.device)
#                 self.Detect.grid[k] = grid    # [1, 1, 80, 80, 2]

#             pred = pred.reshape(bs, self.na * h * w, -1)    # （bs, 80x80, -1）
#             pred_scale = pred.clone()   # clone

#             # de-scale to img size
#             xy_shift = grid.view(1, -1, 2)  # [1, 8400, 2])  grid_xy
#             pred[..., :2] = (pred[..., :2] + xy_shift) * self.stride[k]     # xy
#             pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.stride[k]     # wh

#             # stride between grid 
#             expanded_stride = torch.full((1, xy_shift.shape[1], 1), self.stride[k], device=self.device)     #[1, 6400, 1]

#             # append to list
#             xy_shifts.append(xy_shift)
#             expanded_strides.append(expanded_stride)
#             preds_new.append(pred)              # [[16, 6400, 85], [16, 1600, 85], [16, 400, 85]]
#             preds_scale.append(pred_scale)      # [[16, 6400, 85], [16, 1600, 85], [16, 400, 85]]

#         # concat
#         xy_shifts = torch.cat(xy_shifts, 1)                 # [1, n_anchors_all(8400), 2]
#         expanded_strides = torch.cat(expanded_strides, 1)   # [1, n_anchors_all(8400), 1]
#         preds_scale = torch.cat(preds_scale, 1)             # [16, 8400, 85]
#         p = torch.cat(preds_new, 1)                     # [16, 8400, 85]
#         self.ng = p.shape[1]      # 80x80 + 40x40 + 20x20

#         pbox = p[:, :, :4]                  # at input size. [batch, n_anchors_all, 4]
#         pbox0 = preds_scale[:, :, :4]       # at scales, for l1 loss compute. [batch, n_anchors_all, 4]
#         pobj = p[:, :, 4].unsqueeze(-1)     # [batch, n_anchors_all, 1]
#         pcls = p[:, :, 5: self.no_det]      # [batch, n_anchors_all, n_cls]
        
#         # pick preds mask coefficients
#         pseg = p[:, :, self.no_det: self.no]  # [batch, n_anchors_all(8400), nm32]

#         # return p, pbox, pbox0, pobj, pcls, pkpt, pseg, xy_shifts, expanded_strides
#         return p, pbox, pbox0, pobj, pcls, pseg, xy_shifts, expanded_strides


#     # build targets
#     def build_targets(self, p, targets):
#         # pred => p[0]: {(bs, 1, 80, 80, 85+ns=117), ...}
#         # targets => { num_object, 6(idx, cls, xywh)}

#         # build predictions
#         (   p,                          # [bs, 1, 80, 80, no] => [bs, 8400, no]
#             pbox,                       # [batch, n_anchors_all, 4]
#             pbox0,                      # [batch, n_anchors_all, 4]
#             pobj,                       # [batch, n_anchors_all, 1]
#             pcls,                       # [batch, n_anchors_all, n_cls]
#             pseg,                       # [batch, n_anchors_all, ns(32)]
#             self.xy_shifts,             # [1, n_anchors_all(8400), 2]
#             self.expanded_strides,      # [1, n_anchors_all(8400), 1] 
#         ) = self.build_preds(p)


#         # build targets
#         targets_list = np.zeros((p.shape[0], 1, 5)).tolist()   # batch size
#         for i, item in enumerate(targets.cpu().numpy().tolist()):
#             targets_list[int(item[0])].append(item[1:])
#         max_len = max((len(l) for l in targets_list))
#         empty_list = [[-1] + [0] * 4]  # cls, xywh
#         targets = torch.from_numpy(np.array(list(map(lambda l:l + empty_list * (max_len - len(l)), targets_list)))[:,1:,:]).to(self.device)
#         nts = (targets.sum(dim=2) > 0).sum(dim=1)  # number of objects list per batch [13, 4, 2, ...]

#         # targets cls, box, ...
#         tcls, tbox, tbox_l1, tobj, finalists_masks, num_finalists = [], [], [], [], [], 0 

#         # batch images loop
#         for idx in range(p.shape[0]):   # batch size
#             nt = int(nts[idx])  # num of targets in current image

#             if nt == 0:     # num targets=0  =>  neg sample image
#                 tcls_ = p.new_zeros((0, self.nc))
#                 tbox_ = p.new_zeros((0, 4))
#                 tbox_l1_ = p.new_zeros((0, 4))
#                 tobj_ = p.new_zeros((self.ng, 1))
#                 finalists_mask = p.new_zeros(self.ng).bool()
#                 tseg_ = p.new_zeros((0, self.nm))  # seg
#             else:   
#                 imgsz = torch.Tensor([[self.input_w, self.input_h, self.input_w, self.input_h]]).type_as(targets)  # [[640, 640, 640, 640]]
#                 t_bboxes = targets[idx, :nt, 1: 5].mul_(imgsz)    # gt bbox, de-scaled 
#                 t_classes = targets[idx, :nt, 0]   # gt cls [ 0., 40., 23., 23.]
#                 p_bboxes = pbox[idx]        # pred bbox
#                 p_classes = pcls[idx]       # pred cls
#                 p_objs = pobj[idx]          # pred obj
#                 p_segs = pseg[idx]   # pred mask coefficient

#                 # do label assignment: SimOTA 
#                 (
#                     finalists_mask,
#                     num_anchor_assigned,   
#                     tcls_, 
#                     tobj_, 
#                     tbox_, 
#                     tbox_l1_,
#                  ) = self.get_assignments(p_bboxes, p_classes, p_objs, t_bboxes, t_classes)  
                
#                 # num of assigned anchors in one batch
#                 num_finalists += num_anchor_assigned    

#             # append to list
#             tcls.append(tcls_)
#             tbox.append(tbox_)
#             tobj.append(tobj_)
#             tbox_l1.append(tbox_l1_)
#             finalists_masks.append(finalists_mask)

#         # concat
#         tcls = torch.cat(tcls, 0)
#         tbox_per_batch = tbox   # mini-batch
#         tbox = torch.cat(tbox, 0)
#         tobj = torch.cat(tobj, 0)
#         tbox_l1 = torch.cat(tbox_l1, 0)
#         finalists_masks_per_batch = finalists_masks   # [[8400], [8400], ...]  batch finalists masks, not concat
#         finalists_masks = torch.cat(finalists_masks, 0)
#         num_finalists = max(num_finalists, 1)

#         return pbox, pbox0, pobj, pcls, pseg, tcls, tbox, tbox_l1, tobj, finalists_masks, num_finalists, finalists_masks_per_batch, tbox_per_batch


#     # SimOTA
#     @torch.no_grad()
#     # def get_assignments(self, p_bboxes, p_classes, p_objs, t_bboxes, t_classes, t_kpts=None, p_kpts=None):
#     def get_assignments(self, p_bboxes, p_classes, p_objs, t_bboxes, t_classes):

#         num_objects = t_bboxes.shape[0]   # number of gt object per image
#         candidates_mask, is_in_boxes_and_center = self.get_candidates(t_bboxes)

#         # 2. pick preds in fixed center region, and get bbox, cls, obj
#         p_bboxes = p_bboxes[candidates_mask]
#         cls_preds_ = p_classes[candidates_mask]
#         obj_preds_ = p_objs[candidates_mask]
#         num_in_boxes_anchor = p_bboxes.shape[0]

#         # not neccessary!
#         # if p_segs is not None:
#         #     p_segs = p_segs[candidates_mask]   # [num, nm=32]


#         # 3. iou loss => iou(gts, preds), for calculate dynamic_k
#         pair_wise_ious = pairwise_bbox_iou(t_bboxes, p_bboxes, box_format='xywh')
#         pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

#         # 4. cls loss = cls * obj
#         gt_cls_per_image = (F.one_hot(t_classes.to(torch.int64), self.nc)
#                             .float()
#                             .unsqueeze(1)
#                             .repeat(1, num_in_boxes_anchor, 1))   # gt classes to one hot

#         with torch.cuda.amp.autocast(enabled=False):
#             cls_preds_ = (cls_preds_.float().sigmoid_().unsqueeze(0).repeat(num_objects, 1, 1) 
#                           * obj_preds_.float().sigmoid_().unsqueeze(0).repeat(num_objects, 1, 1))

#             pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
#         del cls_preds_, obj_preds_

#         # 5. cost ==> cls * 1.0 + iou * 3.0 + neg * 1e+5
#         cost = (1.0 * pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 10000.0 * (~is_in_boxes_and_center))  
#         del pair_wise_ious_loss, pair_wise_cls_loss

#         # 6. assign different k positive samples for every gt.
#         (   
#             num_anchor_assigned,
#             pred_ious_this_matching,
#             matched_gt_inds,
#             finalists_mask     
#         ) = self.dynamic_k_matching(cost, pair_wise_ious, t_classes, candidates_mask)
#         del cost, pair_wise_ious

#         # 7. empty cuda cache
#         torch.cuda.empty_cache() 

#         # 8. has anchor point assigned
#         if num_anchor_assigned > 0:
#             # tcls, tbox, tobj
#             tcls_ = t_classes[matched_gt_inds]
#             tcls_ = F.one_hot(tcls_.to(torch.int64), self.nc) * pred_ious_this_matching.unsqueeze(-1)
#             tobj_ = finalists_mask.unsqueeze(-1)  * 1.0
#             tbox_ = t_bboxes[matched_gt_inds]

#             # tbox_l1, do scale
#             tbox_l1_ = p_bboxes.new_zeros((num_anchor_assigned, 4))
#             stride_ = self.expanded_strides[0][finalists_mask]
#             grid_ = self.xy_shifts[0][finalists_mask]
#             tbox_l1_[:, :2] = t_bboxes[matched_gt_inds][:, :2] / stride_ - grid_
#             tbox_l1_[:, 2:4] = torch.log(t_bboxes[matched_gt_inds][:, 2:4] / stride_ + 1e-8)

#         return finalists_mask, num_anchor_assigned, tcls_, tobj_, tbox_, tbox_l1_



#     # get candidates: a fixed center region
#     def get_candidates(self, t_bboxes):
#         num_object = t_bboxes.shape[0]  # number of objects
#         grids_stride = self.expanded_strides[0]  # grid stride
#         grids_xy = self.xy_shifts[0] * grids_stride  # grid coords in each scale: [0, 1, 2, ...] => [0, 8, 16, ...]
        
#         # grid center xy coords
#         grids_center = (grids_xy + 0.5 * grids_stride).unsqueeze(0).repeat(num_object, 1, 1) # [n_anchor, 2] -> [n_gt, n_anchor, 2]

#         # gt top-left & bottom-right coords
#         t_bboxes_tl = (t_bboxes[:, 0:2] - 0.5 * t_bboxes[:, 2:4]).unsqueeze(1).repeat(1, self.ng, 1)
#         t_bboxes_br = (t_bboxes[:, 0:2] + 0.5 * t_bboxes[:, 2:4]).unsqueeze(1).repeat(1, self.ng, 1) # [n_gt, 2] -> [n_gt, n_anchor, 2]
        
#         # check if grid's center is in gt's box
#         bbox_deltas = torch.cat([grids_center - t_bboxes_tl, t_bboxes_br - grids_center], 2)
#         is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
#         is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

#         # in fixed region
#         # TODO: x percentage of width or height, rather than fixed value: 2.5 
#         center_radius = 2.5   # strides=[8, 16, 32] ; 2.5 * strides=[20, 40, 80] = grid_size! that's why 2.5 is better
#         t_bboxes_tl = (t_bboxes[:, 0:2]).unsqueeze(1).repeat(1, self.ng, 1) - center_radius * grids_stride.unsqueeze(0)
#         t_bboxes_br = (t_bboxes[:, 0:2]).unsqueeze(1).repeat(1, self.ng, 1) + center_radius * grids_stride.unsqueeze(0)

#         # check if grid's center is in fixed region
#         center_deltas = torch.cat([grids_center - t_bboxes_tl, t_bboxes_br - grids_center], 2)
#         is_in_centers = center_deltas.min(dim=-1).values > 0.0
#         is_in_centers_all = is_in_centers.sum(dim=0) > 0

#         # in gt boxes or in centers region
#         is_in_boxes_or_center = is_in_boxes_all | is_in_centers_all

#         # in gt boxes and in centers region
#         is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_or_center] & is_in_centers[:, is_in_boxes_or_center]
        
#         return is_in_boxes_or_center, is_in_boxes_and_center


#     # assign different k positive samples for every gt 
#     def dynamic_k_matching(self, cost, pair_wise_ious, t_classes, candidates_mask):

#         torch.cuda.empty_cache()    # empty cuda cache at start
#         ious_in_boxes_matrix = pair_wise_ious   # iou matrix 

#         # get top-k(10) iou anchor point
#         n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
#         topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        
#         # calc dynamic-K for each gt
#         dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
#         dynamic_ks = dynamic_ks.tolist()

#         # create a matching matrix for gt and anchor point(row: gt, col: anchor)
#         matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        
#         # assign num of K anchor points for each gt, based on cost matrix
#         for gt_idx in range(t_classes.shape[0]):   # number of gt objects
#             _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
#             matching_matrix[gt_idx][pos_idx] = 1
#         del topk_ious, dynamic_ks, pos_idx

#         # deal with conflict: filter out the anchor point has been assigned to many gts
#         anchor_matching_gt = matching_matrix.sum(0)
#         if (anchor_matching_gt > 1).sum() > 0:
#             _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
#             matching_matrix[:, anchor_matching_gt > 1] *= 0
#             matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

#         #----------------------------------------------------------------------------------------------------------------
#         # simota enhancement(many2one): assign 1 anchor for gts those have no assigned anchor caused by solving conflict  
#         #----------------------------------------------------------------------------------------------------------------
#         # if (matching_matrix.sum(1) == 0).sum() > 0:    # some gt(row) has no assigned anchors
#         #     cost_non_assigned = cost[matching_matrix.sum(1) == 0, :]
#         #     cols_assigned = matching_matrix.sum(0) > 0
#         #     cost_non_assigned[:, matching_matrix.sum(0) > 0] = 1E10  

#         #     # do linear sum assignment 
#         #     cost_non_assigned_cpu = cost_non_assigned.cpu().numpy()
#         #     i, j = linear_sum_assignment(cost_non_assigned_cpu)

#         #     # create matching matrix non assigned
#         #     matching_matrix_non_assigned = torch.zeros_like(cost_non_assigned, dtype=torch.uint8)
#         #     matching_matrix_non_assigned[i, j] = 1

#         #     # update matching_matrix
#         #     matching_matrix[matching_matrix.sum(1) == 0, :] = matching_matrix_non_assigned


#         # get the number of anchor points which have been assigned to all gts
#         fg_mask_inboxes = matching_matrix.sum(0) > 0
#         num_anchor_assigned = fg_mask_inboxes.sum().item()

#         # update candidates_mask
#         candidates_mask[candidates_mask.clone()] = fg_mask_inboxes
#         matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

#         # matching_matrix * iou matrix
#         pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

#         # finalists_mask
#         finalists_mask = candidates_mask

#         return num_anchor_assigned, pred_ious_this_matching, matched_gt_inds, finalists_mask




            # exit()
            # _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            # matching_matrix[:, anchor_matching_gt > 1] *= 0
            # matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        #--------------------------------------------------------------------------

        # # get the number of anchor points which have been assigned to all gts
        # fg_mask_inboxes = matching_matrix.sum(0) > 0
        # num_anchor_assigned = fg_mask_inboxes.sum().item()

        # # update candidates_mask
        # candidates_mask[candidates_mask.clone()] = fg_mask_inboxes
        # matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        # # matching_matrix * iou matrix
        # pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]

        # # finalists_mask
        # finalists_mask = candidates_mask

        # return num_anchor_assigned, pred_ious_this_matching, matched_gt_inds, finalists_mask


