import os
import numpy as np
import random

import PIL.Image as Image
import torch
import torchvision.datasets
import torchvision.transforms
import torchvision.transforms.functional as func

CLASSES = (
    'no object',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)


class CocoObjectDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root, ann_file, max_len=100, is_train=True):
        super(CocoObjectDetection, self).__init__(root, ann_file)
        self.max_len = max_len
        self.input_size_w = 608
        self.input_size_h = 608
        self.is_train = is_train
        self.cat_id_to_label_map = dict({cat_id: (i + 1) for i, cat_id in enumerate(self.coco.getCatIds(catNms=CLASSES))})  # +1 to pass "no object"

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        img = Image.open(os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])).convert('RGB')

        bboxes = torch.tensor([obj['bbox'] for obj in target])  # (x_min, y_min, w, h)
        category_ids = [obj['category_id'] for obj in target]
        labels = torch.tensor([self.cat_id_to_label_map[category_id] for category_id in category_ids])

        if bboxes.shape[0] == 0:
            bboxes = torch.tensor([[0., 0., 0., 0.]])

        img, pad_mask, bboxes = self._transform(img, bboxes, self.input_size_w, self.input_size_h)
        bboxes, labels = self._pad_with_no_object(bboxes, labels)

        return img, pad_mask, bboxes, labels

    def _pad_with_no_object(self, bboxes, labels):
        _bboxes = torch.zeros((self.max_len, 4))
        _labels = torch.zeros(self.max_len)

        if bboxes.shape[0] != 0 and labels.shape[0] != 0:
            _bboxes[:bboxes.shape[0], :] = bboxes
            _labels[:labels.shape[0]] = labels

        return _bboxes, _labels

    def _transform(self, img, bboxes, input_size_w, input_size_h):
        is_empty = False
        if bboxes.shape[0] == 0:
            is_empty = True
            bboxes = torch.tensor([[0., 0., 0., 0.]])

        bboxes[:, 2:] += bboxes[:, :2]  # (x_min, y_min, w, h) -> (x_min, y_min, x_max, y_max)
        bboxes[:, 0::2].clamp(min=0., max=img.size[0])
        bboxes[:, 1::3].clamp(min=0., max=img.size[1])

        img, bboxes = self._resize(img, bboxes, input_size_w, input_size_h)

        if self.is_train:
            img, bboxes = self._random_hflip(img, bboxes, p=0.5)
            img = torchvision.transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4)(img)

        img, bboxes, pad_mask = self._pad(img, bboxes, input_size_w, input_size_h)
        bboxes = self._normalize_bboxes(bboxes, iw=img.size[0], ih=img.size[1])

        bboxes[:, 2:] -= bboxes[:, :2]  # (x_min, y_min, x_max, y_max) -> (x_min, y_min, w, h)
        bboxes[:, :2] += (bboxes[:, 2:] / 2)  # [x_min, y_min, w, h] -> [cx, cy, w, h]

        if is_empty:
            bboxes = torch.tensor([])

        img = func.to_tensor(img)
        img = func.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, pad_mask, bboxes

    @staticmethod
    def _resize(img, bboxes, input_size_w, input_size_h):
        iw, ih = img.size
        scale = min(input_size_w / iw, input_size_h / ih)
        sh, sw = int(ih * scale), int(iw * scale)

        img = func.resize(img, size=(sh, sw))
        bboxes *= scale

        return img, bboxes

    @staticmethod
    def _random_hflip(img, bboxes, p=0.5):
        if random.random() > p:
            return img, bboxes

        img = func.hflip(img)
        w, _ = img.size

        bboxes = bboxes[:, [2, 1, 0, 3]] * torch.tensor([-1., 1., -1., 1.]) + torch.tensor([w, 0., w, 0.])  # (x_min, y_min, x_max, y_max) -> (-x_max + w, y_min, -x_min + w, y_max)

        return img, bboxes

    @staticmethod
    def _pad(img, bboxes, input_size_w, input_size_h):
        iw, ih = img.size

        pad_width = (input_size_w - iw) / 2
        pad_height = (input_size_h - ih) / 2

        pad_left, pad_right = int(np.floor(pad_width)), int(np.ceil(pad_width))
        pad_top, pad_bottom = int(np.floor(pad_height)), int(np.ceil(pad_height))

        bboxes[:, [0, 2]] += pad_left
        bboxes[:, [1, 3]] += pad_top

        img = func.pad(img, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))
        pad_mask = torch.ones((input_size_h, input_size_w)).bool()
        pad_mask[pad_top:(input_size_h - pad_bottom), pad_left:(input_size_w - pad_right)] = False

        return img, bboxes, pad_mask

    @staticmethod
    def _normalize_bboxes(bboxes, iw, ih):
        bboxes[:, [0, 2]] /= iw  # normalize width
        bboxes[:, [1, 3]] /= ih  # normalize height
        return bboxes
