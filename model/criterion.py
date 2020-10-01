import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy.optimize import linear_sum_assignment
from utils.bbox_ops import bbox_iou


class SetCriterion(nn.Module):
    def __init__(self, matcher, no_object_coefficient):
        super(SetCriterion, self).__init__()
        self.matcher = matcher
        self.no_object_coefficient = no_object_coefficient

    def forward(self, logist_pred, bboxes_pred, classes_gt, bboxes_gt):
        matching_classes_gt, matching_bboxes_gt = self.matcher(logist_pred, bboxes_pred, classes_gt, bboxes_gt)

        label_loss = self.get_label_loss(logist_pred, matching_classes_gt)
        print(label_loss.shape)
        exit(-1)

        return

    def get_label_loss(self, logist_pred, matching_classes_gt):
        no_object_weight = torch.ones(matching_classes_gt.shape)
        no_object_weight[matching_classes_gt == 0] = self.no_object_coefficient

        ce_logist_pred = logist_pred.flatten(start_dim=0, end_dim=1)
        ce_matching_classes_gt = matching_classes_gt.flatten(start_dim=0, end_dim=1)
        ce_no_object_weight = no_object_weight.flatten(start_dim=0, end_dim=1)

        label_loss = func.cross_entropy(input=ce_logist_pred, target=ce_matching_classes_gt, weight=ce_no_object_weight)

        return label_loss


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight=1., bbox_weight=1., giou_weight=1.):
        super(HungarianMatcher, self).__init__()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

    @torch.no_grad()
    def forward(self, logist_pred, bboxes_pred, classes_gt, bboxes_gt):
        # logist_pred: [batch_size, num_queries, num_classes]
        # bboxes_pred: [batch_size, num_queries, 4]
        # classes_gt:  [batch_size, num_queries]
        # bboxes_gt:   [batch_size, num_queries, 4]

        batch_size, num_queries, num_classes = logist_pred.shape

        softmax_logist_pred = logist_pred.softmax(-1)
        classes_gt_one_hot = func.one_hot(classes_gt.long(), num_classes).float()  # [batch_size, num_queries, num_classes]

        cost_label = -torch.bmm(softmax_logist_pred, classes_gt_one_hot.transpose(dim0=1, dim1=2))  # [batch_size, num_queries, num_queries]
        cost_bbox = torch.cdist(bboxes_pred, bboxes_gt, p=1)  # [batch_size, num_queries, num_queries]
        cost_giou = bbox_iou(bboxes_pred, bboxes_gt)  # [batch_size, num_queries, num_queries]

        matching_cost = self.class_weight * cost_label + self.bbox_weight * cost_bbox + self.giou_weight * cost_giou
        matching_cost = matching_cost.cpu().numpy()  # [batch_size, num_queries, num_queries]

        real_object_numbers = classes_gt.bool().sum(dim=-1).cpu().numpy()  # store the real object numbers of the images in the batch

        matching_classes_gt = torch.zeros((batch_size, num_queries))  # same shape as "classes_gt"
        matching_bboxes_gt = torch.zeros((batch_size, num_queries, 4))  # same shape as "bboxes_gt"
        for i in range(batch_size):
            # "pred_idx" is the same shape as "gt_idx", whose length is "real_object_numbers[i]"
            pred_idx, gt_idx = linear_sum_assignment(matching_cost[i, :, :real_object_numbers[i]])
            pred_idx = torch.as_tensor(pred_idx).long()
            gt_idx = torch.as_tensor(gt_idx).long()

            matching_classes_gt[i, pred_idx] = classes_gt.cpu()[i, gt_idx]
            matching_bboxes_gt[i, pred_idx, :] = bboxes_gt.cpu()[i, gt_idx, :]

        return matching_classes_gt.long().to(device=logist_pred.device), matching_bboxes_gt.to(device=logist_pred.device)