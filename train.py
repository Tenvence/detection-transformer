import warnings
import random

import tqdm
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import torchvision.models as cv_models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pycocotools.cocoeval import COCOeval

from datasets.coco_object_detection import CocoObjectDetection, CLASSES
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

    train_dataset_root_path = '../../DataSet/COCO/train2014'
    train_dataset_ann_file = '../../DataSet/COCO/annotations/instances_train2014.json'

    val_dataset_root_path = '../../DataSet/COCO/val2014'
    val_data_ann_file = '../../DataSet/COCO/annotations/instances_val2014.json'

    writer = SummaryWriter(log_dir='./output/log/', comment='detr')
    save_param_file = './output/detr_model.pth'

    train_dataset = CocoObjectDetection(root=train_dataset_root_path, ann_file=train_dataset_ann_file, max_len=num_queries, is_train=True)
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

    # val_dataset = CocoObjectDetection(root=val_dataset_root_path, ann_file=val_data_ann_file, max_len=100, is_train=False)
    # val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True).children())[:-2])
    transformer = Transformer(d_model=512, nhead=8, num_encoders=6, num_decoders=6, dim_feedforward=2048, dropout=0.1)
    detr = Detr(backbone, transformer, num_channels=2048, num_classes=len(CLASSES), num_queries=num_queries).train()

    detr = nn.DataParallel(detr).cuda()

    optimized_parameters = [
        {'params': [p for n, p in detr.module.named_parameters() if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in detr.module.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': base_lr / 10}
    ]
    optimizer = optim.AdamW(optimized_parameters, lr=base_lr, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200)

    matcher = HungarianMatcher(class_weight=1., bbox_weight=5., giou_weight=2.)
    criterion = SetCriterion(matcher, no_object_coef=0.1, label_loss_coef=1., bbox_l1_loss_coef=5., giou_loss_coef=2.)

    # scaler = amp.GradScaler()
    idx = 0
    for epoch in range(max_epoch):
        processor = tqdm.tqdm(train_data_loader)
        losses = []
        for img, pad_mask, bboxes, labels in processor:
            img = img.cuda(non_blocking=True)
            pad_mask = pad_mask.cuda(non_blocking=True)
            # bboxes = bboxes.cuda(non_blocking=True).half()  # for amp
            bboxes = bboxes.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()
            # with amp.autocast():
            #     logist_pred, bboxes_pred = detr(img, pad_mask)
            #     loss = criterion(logist_pred, bboxes_pred, labels, bboxes)
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(detr.parameters(), max_norm=0.1)
            # scaler.step(optimizer)
            # scaler.update()
            logist_pred, bboxes_pred = detr(img, pad_mask)
            loss = criterion(logist_pred, bboxes_pred, labels, bboxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detr.parameters(), max_norm=0.1)
            optimizer.step()

            lr_scheduler.step()

            cur_loss = loss.detach().cpu().numpy()
            losses.append(cur_loss)
            avg_loss = sum(losses) / len(losses)

            writer.add_scalar('Loss/loss', cur_loss, idx)
            writer.add_scalar('Loss/avg_loss', avg_loss, idx)
            idx += 1

            processor.set_description('Epoch: %d / %d, cur_loss: %.4f, avg_loss: %.4f' % (epoch + 1, max_epoch, cur_loss, avg_loss))

        torch.save(detr.module.state_dict(), save_param_file)


if __name__ == '__main__':
    main()
