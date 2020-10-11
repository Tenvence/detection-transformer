import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as cv_models
import tqdm
from torch.utils.data import DataLoader

from datasets.coco_object_detection import CocoObjectDetection
from model.detr import Detr
from model.transformer import Transformer
from model.criterion import HungarianMatcher, SetCriterion


def main():
    dataset_root_path = '../../DataSet/COCO/train2014'
    dataset_ann_file = '../../DataSet/COCO/annotations/instances_train2014.json'

    dataset = CocoObjectDetection(root=dataset_root_path, ann_file=dataset_ann_file, max_len=100)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16)

    backbone = nn.Sequential(*list(cv_models.resnet50(pretrained=True).children())[:-2])
    transformer = Transformer(d_model=512, nhead=8, num_encoders=6, num_decoders=6, dim_feedforward=2048, dropout=0.1)
    detr = Detr(backbone, transformer, num_channels=2048, num_classes=81, num_queries=100)
    detr = nn.DataParallel(detr).cuda().train()

    optimized_parameters = [
        {'params': [p for n, p in detr.module.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': 1e-4},
        {'params': [p for n, p in detr.module.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': 1e-5}
    ]
    optimizer = optim.AdamW(optimized_parameters, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200)

    matcher = HungarianMatcher(class_weight=1., bbox_weight=5., giou_weight=2.)
    criterion = SetCriterion(matcher, no_object_coef=0.5, label_loss_coef=1., bbox_l1_loss_coef=5., giou_loss_coef=2.)
    max_epoch = 300

    processor = tqdm.tqdm(data_loader)
    for i in range(max_epoch):
        for img, pad_mask, bboxes, labels in processor:
            img, pad_mask, bboxes, labels = img.cuda(), pad_mask.cuda(), bboxes.cuda(), labels.cuda()
            logist_pred, bboxes_pred = detr(img, pad_mask)
            loss = criterion(logist_pred, bboxes_pred, labels, bboxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            processor.set_description('Epoch: %d/%d, loss: %.4f' % (i + 1, max_epoch, loss))

        torch.save(detr.module.state_dict(), 'detr_model.pth')


if __name__ == '__main__':
    main()
