import torch


def bbox_iou(bboxes1, bboxes2, bbox_type='xywh', return_union=False):
    if bbox_type == 'xywh':
        bboxes1 = convert_bbox_type(bboxes1, target_bbox_type='xyxy')
        bboxes2 = convert_bbox_type(bboxes2, target_bbox_type='xyxy')

    area1 = bbox_area(bboxes1)[:, :, None].repeat(1, 1, bboxes1.shape[1])  # [B, num_queries, num_queries]
    area2 = bbox_area(bboxes2)[:, None, :].repeat(1, bboxes2.shape[1], 1)  # [B, num_queries, num_queries]

    lt = torch.max(bboxes1[:, :, None, :2].repeat(1, 1, bboxes1.shape[1], 1), bboxes2[:, None, :, :2].repeat(1, bboxes2.shape[1], 1, 1))  # [B, num_queries, num_queries, 2]
    rb = torch.min(bboxes1[:, :, None, 2:].repeat(1, 1, bboxes1.shape[1], 1), bboxes2[:, None, :, 2:].repeat(1, bboxes2.shape[1], 1, 1))  # [B, num_queries, num_queries, 2]
    wh = (rb - lt).clamp(min=0.)  # [B, num_queries, num_queries, 2]
    inter = wh[..., 0] * wh[..., 1]  # [B, num_queries, num_queries]
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)

    if return_union:
        return iou, union
    else:
        return iou


def bbox_giou(bboxes1, bboxes2, bbox_type='xywh'):
    iou, union = bbox_iou(bboxes1, bboxes2, bbox_type)

    lt = torch.min(bboxes1[:, :, None, :2].repeat(1, 1, bboxes1.shape[1], 1), bboxes2[:, None, :, :2].repeat(1, bboxes2.shape[1], 1, 1))  # [B, num_queries, num_queries, 2]
    rb = torch.max(bboxes1[:, :, None, 2:].repeat(1, 1, bboxes1.shape[1], 1), bboxes2[:, None, :, 2:].repeat(1, bboxes2.shape[1], 1, 1))  # [B, num_queries, num_queries, 2]
    wh = (rb - lt).clamp(min=0.)  # [B, num_queries, num_queries, 2]
    closure = wh[..., 0] * wh[..., 1]

    return iou - (closure - union) / (closure + 1e-7)


def convert_bbox_type(bboxes, target_bbox_type='xyxy'):
    if target_bbox_type == 'xyxy':
        bboxes[..., :2] -= bboxes[..., 2:] / 2  # cx -> cx - w/2 -> x_min; cy -> cy - h/2 -> y_min
        bboxes[..., 2:] += bboxes[..., :2]  # w -> w + x_min -> x_max; h -> h + y_min -> y_max
    else:
        bboxes[..., 2:] -= bboxes[..., :2]  # x_max -> x_max - x_min -> w; y_max -> y_max - y_min -> h
        bboxes[..., :2] += bboxes[..., 2:] / 2  # x_min -> x_min + w/2 -> x_max; y_min -> y_min + h/2 -> y_max

    return bboxes


def bbox_area(bboxes):
    wh = bboxes[..., 2:] - bboxes[..., :2]
    area = wh[..., 0] * wh[..., 1]
    return area
