import json
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.models as cv_models
import tqdm
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.coco_object_detection_train import CocoObjectDetectionTrain
from datasets.coco_object_detection_test import CocoObjectDetectionTest
from datasets.coco_tools import CLASSES
from model.criterion import HungarianMatcher, SetCriterion
from model.detr import Detr
from model.transformer import Transformer

warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


def main():
    max_epoch = 300
    base_lr = 1e-4
    num_queries = 100
    input_size_h, input_size_w = 512, 512

    train_dataset_root_path = '../../DataSet/COCO/train2014'
    train_dataset_ann_file = '../../DataSet/COCO/annotations/instances_train2014.json'

    val_dataset_root_path = '../../DataSet/COCO/val2014'
    val_data_ann_file = '../../DataSet/COCO/annotations/instances_val2014.json'

    writer = SummaryWriter(log_dir='./output/log/')
    save_param_file = './output/detr_model.pth'
    checkpoint_file_prefix = './output/checkpoint_epoch_%d.pth'
    recover_checkpoint_epoch = None
    val_res_file_prefix = './output/val_epoch_%d.json'

    train_dataset = CocoObjectDetectionTrain(root=train_dataset_root_path, ann_file=train_dataset_ann_file, input_size=(input_size_h, input_size_w), max_len=num_queries)
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

    # val_dataset = CocoObjectDetectionTest(root=val_dataset_root_path, ann_file=val_data_ann_file, input_size=(input_size_h, input_size_w))
    # val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True).children())[:-2])
    transformer = Transformer(d_model=512, nhead=8, num_encoders=6, num_decoders=6, dim_feedforward=2048, dropout=0.1)
    detr = Detr(backbone, transformer, num_channels=2048, num_classes=len(CLASSES), num_queries=num_queries)

    detr = nn.DataParallel(detr).cuda()

    optimized_parameters = [
        {'params': [p for n, p in detr.module.named_parameters() if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in detr.module.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': base_lr / 10}
    ]
    optimizer = optim.AdamW(optimized_parameters, lr=base_lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80)

    matcher = HungarianMatcher(class_weight=1., bbox_weight=5., giou_weight=2.)
    criterion = SetCriterion(matcher, no_object_coef=0.1, label_loss_coef=1., bbox_l1_loss_coef=5., giou_loss_coef=2.)

    checkpoint_epoch = None
    if recover_checkpoint_epoch is not None:
        recover_checkpoint = torch.load(checkpoint_file_prefix % recover_checkpoint_epoch)
        detr.module.load_state_dict(recover_checkpoint['model'])
        optimizer.load_state_dict(recover_checkpoint['optimizer'])
        lr_scheduler.load_state_dict(recover_checkpoint['lr_scheduler'])
        checkpoint_epoch = recover_checkpoint['cur_epoch']

    for epoch in range(max_epoch) if recover_checkpoint_epoch is None else range(checkpoint_epoch, max_epoch):
        print('\nEPOCH: %d / %d' % (epoch + 1, max_epoch))
        print('  TRAIN:')
        detr.train()
        train_processor = tqdm.tqdm(train_data_loader)
        losses = []
        for img, pad_mask, bboxes, labels in train_processor:
            img = img.cuda(non_blocking=True)
            pad_mask = pad_mask.cuda(non_blocking=True)
            bboxes = bboxes.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()
            logist_pred, bboxes_pred = detr(img, pad_mask)
            loss = criterion(logist_pred, bboxes_pred, labels, bboxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detr.parameters(), max_norm=0.1)
            optimizer.step()

            cur_loss = loss.detach().cpu().numpy()
            losses.append(cur_loss)
            avg_loss = sum(losses) / len(losses)

            train_processor.set_description('    cur_loss: %.4f, avg_loss: %.4f' % (cur_loss, avg_loss))

        writer.add_scalar('Loss/avg_loss_epoch', sum(losses) / len(losses), epoch)
        lr_scheduler.step()

        if os.path.exists(checkpoint_file_prefix % (epoch - 5)):
            os.remove(checkpoint_file_prefix % (epoch - 5))

        torch.save({
            'model': detr.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'cur_epoch': epoch
        }, checkpoint_file_prefix % epoch)  # save_checkpoint

        # print('  EVAL:')
        # detr.eval()
        # val_processor = tqdm.tqdm(val_data_loader)
        # val_json_dict = []
        # for img, pad_mask, img_id, iw, ih, resize_scale in val_processor:
        #     img = img.cuda(non_blocking=True)
        #     pad_mask = pad_mask.cuda(non_blocking=True)
        #     resize_scale = resize_scale.cuda(non_blocking=True)
        #     iw = iw.cuda(non_blocking=True)
        #     ih = ih.cuda(non_blocking=True)
        #
        #     logist_pred, bboxes_pred = detr(img, pad_mask)
        #     scores, labels = logist_pred.softmax(dim=-1).max(dim=-1)  # [B, num_queries]
        #
        #     bboxes_pred[..., [0, 1]] -= bboxes_pred[..., [2, 3]] / 2  # [norm_cx, norm_cy, norm_w, norm_h] -> [norm_x_min, norm_y_min, norm_w, norm_h]
        #     bboxes_pred[..., [0, 1]] = bboxes_pred[..., [0, 1]].clamp(min=0., max=1.)
        #     bboxes_pred[..., [2, 3]] = torch.where(bboxes_pred[..., [0, 1]] + bboxes_pred[..., [2, 3]] < 1., bboxes_pred[..., [2, 3]], 1. - bboxes_pred[..., [0, 1]])
        #     bboxes_pred[..., [0, 2]] *= iw  # [norm_x_min, norm_y_min, norm_w, norm_h] -> [x_min, norm_y_min, w, norm_h]
        #     bboxes_pred[..., [1, 3]] *= ih  # [x_min, norm_y_min, w, norm_h] -> [x_min, y_min, w, h]
        #     bboxes_pred /= resize_scale[:, None, None]  # remove resize scale
        #
        #     for batch_idx in range(bboxes_pred.shape[0]):
        #         image_id = int(img_id[batch_idx])
        #
        #         object_mask = labels[batch_idx, :] != 0  # [num_queries]
        #         labels_in_img = labels[batch_idx, object_mask]  # [num_objects] filtered by "object_mask"
        #         scores_in_img = scores[batch_idx, object_mask]
        #         bboxes_in_img = bboxes_pred[batch_idx, object_mask, :]  # [num_objects, 4]
        #
        #         num_objects = labels_in_img.shape[0]
        #         val_processor.set_description('    num_objects: %d' % num_objects)
        #
        #         for i in range(num_objects):
        #             label, score, bbox = labels_in_img[i], scores_in_img[i], bboxes_in_img[i, :]
        #             x_min, y_min, w, h = bbox.cpu()
        #             val_json_dict.append({
        #                 'image_id': image_id,
        #                 'category_id': val_dataset.coco.getCatIds(catNms=CLASSES[label.cpu().numpy()])[0],
        #                 'bbox': [float(str('%.1f' % x_min)), float(str('%.1f' % y_min)), float(str('%.1f' % w)), float(str('%.1f' % h))],
        #                 'score': float(str('%.4f' % score.cpu())),
        #             })
        #
        # val_res_file = val_res_file_prefix % epoch
        # with open(val_res_file, 'w') as f:
        #     json.dump(val_json_dict, f)
        #
        # if len(val_json_dict) == 0:
        #     continue
        #
        # coco_evaluator = COCOeval(cocoGt=val_dataset.coco, cocoDt=val_dataset.coco.loadRes(val_res_file), iouType='bbox')
        # coco_evaluator.evaluate()
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        #
        # ap, ap50, ap75, ap_s, ap_m, ap_l, _, _, _, _, _, _ = coco_evaluator.stats
        #
        # writer.add_scalar('Metric/AP', ap, epoch)
        # writer.add_scalar('Metric/AP50', ap50, epoch)
        # writer.add_scalar('Metric/AP75', ap75, epoch)
        # writer.add_scalar('Metric/AP_S', ap_s, epoch)
        # writer.add_scalar('Metric/AP_M', ap_m, epoch)
        # writer.add_scalar('Metric/AP_L', ap_l, epoch)


if __name__ == '__main__':
    main()
