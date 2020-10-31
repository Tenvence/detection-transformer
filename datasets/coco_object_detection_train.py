import os
import random

import PIL.Image as Image
import torch
import torchvision.datasets
import torchvision.transforms
import torchvision.transforms.functional as func

from .coco_tools import CLASSES, resize_img, pad_img, norm_img


class CocoObjectDetectionTrain(torchvision.datasets.CocoDetection):
    def __init__(self, root, ann_file, input_size, max_len=100):
        super(CocoObjectDetectionTrain, self).__init__(root, ann_file)
        self.max_len = max_len
        self.input_size_h, self.input_size_w = input_size
        self.cat_id_to_label_map = dict({cat_id: (i + 1) for i, cat_id in enumerate(self.coco.getCatIds(catNms=CLASSES))})  # +1 to pass "no object"

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        img = Image.open(os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])).convert('RGB')

        bboxes = torch.tensor([obj['bbox'] for obj in target])  # (x_min, y_min, w, h)
        category_ids = [obj['category_id'] for obj in target]
        labels = torch.tensor([self.cat_id_to_label_map[category_id] for category_id in category_ids])

        img, pad_mask, bboxes = self._transform(img, bboxes)
        bboxes, labels = self._pad_with_no_object(bboxes, labels)

        return img, pad_mask, bboxes, labels

    def __len__(self):
        return len(self.ids)

    def _pad_with_no_object(self, bboxes, labels):
        _bboxes = torch.zeros((self.max_len, 4))
        _labels = torch.zeros(self.max_len)

        if bboxes.shape[0] != 0 and labels.shape[0] != 0:
            _bboxes[:bboxes.shape[0], :] = bboxes
            _labels[:labels.shape[0]] = labels

        return _bboxes, _labels

    def _transform(self, img, bboxes):
        is_empty = False
        if bboxes.shape[0] == 0:
            is_empty = True
            bboxes = torch.tensor([[0., 0., 0., 0.]])

        bboxes[:, 2:] += bboxes[:, :2]  # (x_min, y_min, w, h) -> (x_min, y_min, x_max, y_max)
        bboxes[:, 0::2].clamp(min=0., max=img.size[0])
        bboxes[:, 1::3].clamp(min=0., max=img.size[1])

        img, bboxes = self._resize(img, bboxes)
        img, bboxes = self._random_hflip(img, bboxes, p=0.5)
        img = torchvision.transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4)(img)

        bboxes = self._normalize_bboxes(bboxes, nw=img.size[0], nh=img.size[1])
        bboxes[:, 2:] -= bboxes[:, :2]  # (x_min, y_min, x_max, y_max) -> (x_min, y_min, w, h)
        bboxes[:, :2] += (bboxes[:, 2:] / 2)  # [x_min, y_min, w, h] -> [cx, cy, w, h]

        img, pad_mask = pad_img(img, self.input_size_w, self.input_size_h)
        img = norm_img(img)

        if is_empty:
            bboxes = torch.tensor([])

        return img, pad_mask, bboxes

    def _resize(self, img, bboxes):
        img, resize_scale = resize_img(img, self.input_size_w, self.input_size_h)
        bboxes *= resize_scale

        return img, bboxes

    @staticmethod
    def _random_hflip(img, bboxes, p=0.5):
        if random.random() > p:
            return img, bboxes

        img = func.hflip(img)
        w, _ = img.size

        # (x_min, y_min, x_max, y_max) -> (-x_max + w, y_min, -x_min + w, y_max)
        bboxes = bboxes[:, [2, 1, 0, 3]] * torch.tensor([-1., 1., -1., 1.]) + torch.tensor([w, 0., w, 0.])

        return img, bboxes

    @staticmethod
    def _normalize_bboxes(bboxes, nw, nh):
        bboxes[:, [0, 2]] /= nw  # normalize width
        bboxes[:, [1, 3]] /= nh  # normalize height
        return bboxes
