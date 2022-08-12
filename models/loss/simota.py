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




class ComputeLoss:
    # SimOTA
    def __init__(self, model):
        LOGGER.info(f"{colorstr('ComputeLoss: ')} SimOTA")

        self.device = next(model.parameters()).device  # get model device
        self.hyp = model.hyp  # hyperparameters
        self.head = de_parallel(model).model[-1]  # Detect() module
        self.ng = 0   # number of grid in every scale: 80x80 + 40x40 + 20x20

        # head attrs
        for x in ('nl', 'na', 'nc', 'stride', 'nk', 'no_det', 'no_kpt', 'no'):
            setattr(self, x, getattr(self.head, x))
        
        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(reduction="none")   # reduction="mean" default, pos_weights=None
        self.BCEobj = nn.BCEWithLogitsLoss(reduction="none")   # TODO: add pos_weights=None
        self.L1box = nn.L1Loss(reduction="none")
        if self.nk > 0:
            # self.BCEkpt = nn.BCEWithLogitsLoss(reduction="none")  # kpt
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
            self.OKSkpt = OKSLoss(kpts_weights)


    def __call__(self, p, targets):
        # p: {(bs, 1, 80, 80, no), (bs, 1, 40, 40, no), ...}
        # targets: { num_object, 6 + no_kpt(idx, cls, xywh, kpts(optional)) }

        # loss item init
        lcls = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device) 
        lbox_l1 = torch.zeros(1, device=self.device)
        # if self.nk > 0:
        lkpt = torch.zeros(1, device=self.device)

        # build targets
        (   pbox, pbox0, pobj, pcls, pkpt,
            tcls, tbox, tbox_l1, tobj, tkpt,
            finalists_masks, num_finalists
        ) = self.build_targets(p, targets)

        # compute loss
        lbox += (1.0 - bbox_iou(pbox.view(-1, 4)[finalists_masks], tbox, SIoU=True).squeeze()).sum() / num_finalists  # iou(prediction, target)
        lbox_l1 += (self.L1box(pbox0.view(-1, 4)[finalists_masks], tbox_l1)).sum() / num_finalists
        lobj += (self.BCEobj(pobj.view(-1, 1), tobj * 1.0)).sum() / num_finalists
        lcls += (self.BCEcls(pcls.view(-1, self.nc)[finalists_masks], tcls)).sum() / num_finalists
        if self.nk > 0 and pkpt is not None and tkpt is not None:   # kpt loss
            # -------------------------
            #   OKS Loss for kpts  
            #   TODO: Wingloss or SmoothL1 loss, ... for other kpts task 
            # -------------------------
            lkpt += self.OKSkpt(pkpt.view(-1, self.no_kpt)[finalists_masks], tkpt, tbox).sum() / num_finalists


        # loss weighted
        lbox *= self.hyp['box']    # self.hyp.get('box', 5.0)    
        lbox_l1 *= self.hyp['box_l1']    # self.hyp.get('box_l1', 1.0)    
        lbox += lbox_l1
        lcls *= self.hyp['cls']    # self.hyp.get('cls', 1.0)
        lobj *= self.hyp['obj']    # self.hyp.get('obj', 1.0)
        lkpt *= self.hyp['kpt']    # self.hyp.get('kpt', 5.5)    

        return lbox + lobj + lcls + lkpt, torch.cat((lbox, lobj, lcls, lkpt)).detach()  


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
                yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
                grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).to(self.device)
                self.head.grid[k] = grid    # [1, 1, 80, 80, 2]

            pred = pred.reshape(bs, self.na * h * w, -1)    # （bs, 80x80, -1）
            pred_scale = pred.clone()   # clone

            # de-scale to img size
            xy_shift = grid.view(1, -1, 2)  # [1, 8400, 2])  grid_xy
            pred[..., :2] = (pred[..., :2] + xy_shift) * self.stride[k]     # xy
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.stride[k]     # wh

            # kpt
            if self.nk > 0:
                kpt_conf_grids = torch.zeros_like(xy_shift)[..., 0:1]
                kpt_grids = torch.cat((xy_shift, kpt_conf_grids), dim = 2).repeat(1, 1, self.nk)
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

        if self.nk > 0:
            empty_list = [[-1] + [0] * (self.nk * 2 + 4)]  # cls, xy * self.nk 
        else: 
            empty_list = [[-1] + [0] * 4]  # cls, xywh

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

                # keypoint: de-scale to origin image size  !!!!
                if self.nk > 0:
                    imgsz_kpt = torch.Tensor([[input_w, input_h] * self.nk]).type_as(targets)  # [[640, 640, 640, 640]]
                    t_kpts = targets[idx, :nt, -2 * self.nk:].mul_(imgsz_kpt)  # t_kpts
                else:
                    t_kpts = None

                # do label assignment: SimOTA 
                (
                    finalists_mask,
                    num_anchor_assigned,   
                    tcls_, 
                    tobj_, 
                    tbox_, 
                    tbox_l1_,
                    tkpt_
                 ) = self.get_assignments(p_bboxes, p_classes, p_objs, t_bboxes, t_classes, t_kpts)
                
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


        return ( pbox, pbox0, pobj, pcls, pkpt,
                 tcls, tbox, tbox_l1, tobj, tkpt,
                 finalists_masks, num_finalists )


    # SimOTA
    @torch.no_grad()
    def get_assignments(self, p_bboxes, p_classes, p_objs, t_bboxes, t_classes, t_kpts=None):

        num_objects = t_bboxes.shape[0]   # number of gt object per image

        # 1. get candidates: {a fixed center region} + {gt box} 
        candidates_mask, is_in_boxes_and_center = self.get_candidates(t_bboxes)

        # 2. pick preds in fixed center region, and get bbox, cls, obj
        p_bboxes = p_bboxes[candidates_mask]
        cls_preds_ = p_classes[candidates_mask]
        obj_preds_ = p_objs[candidates_mask]
        num_in_boxes_anchor = p_bboxes.shape[0]

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

        # 5. cost
        cost = (1.0 * pair_wise_cls_loss        # 1.0
                + 3.0 * pair_wise_ious_loss     # 3.0
                + 10000.0 * (~is_in_boxes_and_center))     # neg samples, 

        # 6. assign different k positive samples for every gt.
        (   
            num_anchor_assigned,
            pred_ious_this_matching,
            matched_gt_inds,
            finalists_mask     
        ) = self.dynamic_k_matching(cost, pair_wise_ious, t_classes, candidates_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        # 7. empty cuda cache
        torch.cuda.empty_cache() 

        # 8. has anchor point assigned
        if num_anchor_assigned > 0:
            # tcls, tbox, tobj
            tcls_ = t_classes[matched_gt_inds]
            tcls_ = F.one_hot(tcls_.to(torch.int64), self.nc) * pred_ious_this_matching.unsqueeze(-1)
            tobj_ = finalists_mask.unsqueeze(-1)
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
        center_radius = 2.5   # strides=[8, 16, 32] ; 2.5 * strides=[20, 40, 80] = grid_size! that's why set to 2.5
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

        #--------------------------------------------------------------------------
        # Fix simota bug(many2one): assign 1 anchor for gts those have no assigned anchor caused by solving conflict  
        #--------------------------------------------------------------------------
        if (matching_matrix.sum(1) == 0).sum() > 0:    # some gt(row) has no assigned anchors
            cost_non_assigned = cost[matching_matrix.sum(1) == 0, :]
            cols_assigned = matching_matrix.sum(0) > 0
            
            # print(f'cols_assigned: {cols_assigned}')
            idx_assigned = []
            for i, x in enumerate(cols_assigned):
                if x.item() is True:
                    idx_assigned.append(i)    
                    # print(f'col has assigned =====> {i}')
            # print(f'============> idx col assigned:\n {idx_assigned}')

            # assign bigger enough cost value(1e10) for already assigned anchors
            # print(f"==> cost_non_assigned( > 0) {cost_non_assigned[:, matching_matrix.sum(0) > 0]}")
            # print(f"==> cost_non_assigned( == 0) {cost_non_assigned[:, matching_matrix.sum(0) == 0]}")
            # print(f"==> (min) cost_non_assigned {torch.min(cost_non_assigned, dim=0)[0]}")
            # print(f"==> (min) cost_non_assigned {torch.min(cost_non_assigned, dim=0)}")
            
            cost_non_assigned[:, matching_matrix.sum(0) > 0] = 1e+10  
            # print(f"after ==> cost_non_assigned( > 0) {cost_non_assigned[:, matching_matrix.sum(0) > 0]}")
            # print(f"after ==> cost_non_assigned( == 0) {cost_non_assigned[:, matching_matrix.sum(0) == 0]}")
            # for i, row in enumerate(cost_non_assigned):
            #     for j, col in enumerate(row):
            #         if col == 1e+10:
            #             print(f'cost assigned col (1e10) =====> {(i, j)}')



            # print('=====>cost_non_assigned\n', cost_non_assigned)

            # do linear sum assignment 
            cost_non_assigned_cpu = cost_non_assigned.cpu().numpy()
            i, j = linear_sum_assignment(cost_non_assigned_cpu)

            for idx, jj in enumerate(j):
                if jj in idx_assigned:
                    print('===> conflict!!!!! :::::=> ', jj)
                    print('=====> i,j (cost_non_assigned_cpu) ', cost_non_assigned_cpu[i[idx]][j[idx]])
                    print('=====> j (cost) ', cost[:, j[idx]])


            # print(f'linear sum assigned: ===> {i, j}')

            # create matching matrix non assigned
            matching_matrix_non_assigned = torch.zeros_like(cost_non_assigned, dtype=torch.uint8)
            matching_matrix_non_assigned[i, j] = 1

            # print(f'matching_matrix_non_assigned: ===> {matching_matrix_non_assigned}')


            # update matching_matrix
            matching_matrix[matching_matrix.sum(1) == 0, :] = matching_matrix_non_assigned


        # check again if matching matrix still has conflicts
        # TODO: if still has conflicts, re-assign
        assert (matching_matrix.sum(1) == 0).sum() == 0, '>>> Not all GTs have been assigned!'


        # assert (matching_matrix.sum(0) > 1).sum() == 0, '>>> Matching matrix still has conflicts!!!!!!!'
        # deal with conflict: filter out the anchor point has been assigned to many gts
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            print('Matching matrix still has conflicts!!!!!!!')
            print(matching_matrix.sum(0))
            print(matching_matrix.sum(0) > 1)
            print((matching_matrix.sum(0) > 1).sum())

            print(f'===> conflict cols:\n {cost[:, anchor_matching_gt > 1]}')

            # exit()
            # _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            # matching_matrix[:, anchor_matching_gt > 1] *= 0
            # matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        #--------------------------------------------------------------------------

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

