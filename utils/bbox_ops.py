import torch


def bbox_iou(bboxes1, bboxes2, for_pair=True):
    if for_pair:
        area1 = bbox_area(bboxes1)  # [B, num_queries]
        area2 = bbox_area(bboxes2)  # [B, num_queries]

        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, num_queries, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, num_queries, 2]
    else:
        area1 = bbox_area(bboxes1)[:, :, None].repeat(1, 1, bboxes1.shape[1])  # [B, num_queries, num_queries]
        area2 = bbox_area(bboxes2)[:, None, :].repeat(1, bboxes2.shape[1], 1)  # [B, num_queries, num_queries]

        lt = torch.max(bboxes1[:, :, None, :2].repeat(1, 1, bboxes1.shape[1], 1), bboxes2[:, None, :, :2].repeat(1, bboxes2.shape[1], 1, 1))  # [B, num_queries, num_queries, 2]
        rb = torch.min(bboxes1[:, :, None, 2:].repeat(1, 1, bboxes1.shape[1], 1), bboxes2[:, None, :, 2:].repeat(1, bboxes2.shape[1], 1, 1))  # [B, num_queries, num_queries, 2]

    wh = (rb - lt).clamp(min=0.)  # [B, num_queries, num_queries, 2]
    inter = wh[..., 0] * wh[..., 1]  # [B, num_queries, num_queries]
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)

    return iou, union


def bbox_giou(bboxes1, bboxes2, for_pair=True):
    iou, union = bbox_iou(bboxes1, bboxes2, for_pair=for_pair)

    if for_pair:
        lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])  # [B, num_queries, 2]
        rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, num_queries, 2]
    else:
        lt = torch.min(bboxes1[:, :, None, :2].repeat(1, 1, bboxes1.shape[1], 1), bboxes2[:, None, :, :2].repeat(1, bboxes2.shape[1], 1, 1))  # [B, num_queries, num_queries, 2]
        rb = torch.max(bboxes1[:, :, None, 2:].repeat(1, 1, bboxes1.shape[1], 1), bboxes2[:, None, :, 2:].repeat(1, bboxes2.shape[1], 1, 1))  # [B, num_queries, num_queries, 2]

    wh = (rb - lt).clamp(min=0.)  # [B, num_queries, num_queries, 2]
    closure = wh[..., 0] * wh[..., 1]

    return iou - (closure - union) / (closure + 1e-7)


def convert_bbox_xywh_xyxy(bboxes):
    cx, cy, w, h = bboxes.unbind(dim=-1)
    b = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    return torch.stack(b, dim=-1)


def bbox_area(bboxes):
    wh = (bboxes[..., 2:] - bboxes[..., :2]).clamp(min=0.)
    area = wh[..., 0] * wh[..., 1]
    return area
