from typing import List

import torch

from configs import args


class Metric(object):
    def __init__(self):

        if args.dataset_name == 'pascal':
            self.nclass = 20+1
        elif args.dataset_name == 'coco':
            self.nclass = 81
        else:
            raise

        self.intersection_buf = torch.zeros([2, self.nclass]).float().to(args.device)
        self.union_buf = torch.zeros([2, self.nclass]).float().to(args.device)
        self.ones = torch.ones_like(self.union_buf)

    def update(self, pred_mask, gt_mask, gt_ignore, class_id:List[int]):
        class_id = torch.Tensor(class_id).int().to(args.device).repeat(len(gt_mask))
        inter_b, union_b = self.classify_prediction(pred_mask, gt_mask, gt_ignore)
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, torch.where(self.union_buf[1]!=0)[0])
        miou = iou[1].mean()

        fb_iou = (self.intersection_buf.index_select(1, torch.where(self.union_buf[1]!=0)[0]).sum(dim=1) /
                  self.union_buf.index_select(1, torch.where(self.union_buf[1]!=0)[0]).sum(dim=1)).mean()

        return miou, fb_iou

    def classify_prediction(self, pred_mask, gt_mask, gt_ignore=None, ignore_index=255):
        if gt_ignore is not None:
            assert torch.logical_and(gt_ignore, gt_mask).sum() == 0
            gt_ignore = ignore_index * gt_ignore.int()
            gt_mask = gt_mask.int() + gt_ignore
            pred_mask[gt_mask == ignore_index] = ignore_index

        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.tensor([(_inter==0).sum(), (_inter==1).sum()], device=_pred_mask.device).int()
            area_inter.append(_area_inter)
            area_pred.append(torch.tensor([(_pred_mask==0).sum(), (_pred_mask==1).sum()], device=_pred_mask.device).int())
            area_gt.append(torch.tensor([(_gt_mask==0).sum(), (_gt_mask==1).sum()], device=_pred_mask.device).int())
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union
    

