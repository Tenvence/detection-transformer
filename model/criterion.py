import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy.optimize import linear_sum_assignment
from utils.bbox_ops import bbox_iou


class SetCriterion(nn.Module):
    def __init__(self):
        super(SetCriterion, self).__init__()

    def forward(self, x):
        return

    def label_loss(self, output, target):
        pass


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight=1., bbox_weight=1., giou_weight=1.):
        super(HungarianMatcher, self).__init__()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

    @torch.no_grad()
    def forward(self, logist_pred, bboxes_pred, classes_gt, bboxes_gt):
        logist_pred = logist_pred.softmax(-1)
        classes_gt_one_hot = func.one_hot(classes_gt.long(), num_classes=logist_pred.shape[-1]).float()

        cost_label = -torch.bmm(logist_pred, classes_gt_one_hot.transpose(dim0=1, dim1=2))
        cost_bbox = torch.cdist(bboxes_pred, bboxes_gt, p=1)
        cost_giou = bbox_iou(bboxes_pred, bboxes_gt)

        matching_cost = self.class_weight * cost_label + self.bbox_weight * cost_bbox + self.giou_weight * cost_giou
        matching_cost = matching_cost.cpu().numpy()
        no_object_pad_count = classes_gt.bool().sum(dim=-1).cpu().numpy()
        indices = [linear_sum_assignment(matching_cost[i, :, :no_object_pad_count[i]]) for i in range(matching_cost.shape[0])]

        return [(torch.as_tensor(i).long(), torch.as_tensor(j).long()) for i, j in indices]
