import os
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
    def __init__(self, root, ann_file, max_len=100):
        super(CocoObjectDetection, self).__init__(root, ann_file)
        self.max_len = max_len

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        img = Image.open(os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])).convert('RGB')

        bboxes = torch.tensor([obj['bbox'] for obj in target])  # (x, y, w, h)
        category_ids = [obj['category_id'] for obj in target]
        labels = torch.tensor([self._map_category_id_to_label(category_id) for category_id in category_ids])

        bboxes, labels = self._pad_with_no_object(bboxes, labels)
        bboxes[:, 2:] += bboxes[:, :2]  # (x, y, w, h) -> (x_min, y_min, x_max, y_max)

        img, bboxes = self._transform(img, bboxes)

        return img, bboxes, labels

    def _map_category_id_to_label(self, category_id):
        return dict({cat_id: i for i, cat_id in enumerate(self.coco.getCatIds(catNms=CLASSES))})[category_id]

    def _transform(self, img, bboxes):
        if random.random() > 0.5:
            img, bboxes = self._hflip(img, bboxes)

        img, bboxes = self._resize_pad(img, bboxes, input_size_w=608, input_size_h=608)
        img = torchvision.transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4)(img)

        iw, ih = img.size
        bboxes[:, [0, 2]] /= iw
        bboxes[:, [1, 3]] /= ih
        bboxes[:, 2:] -= bboxes[:, :2]  # (x_min, y_min, x_max, y_max) -> (x, y, w, h)

        img = func.to_tensor(img)
        img = func.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, bboxes

    @staticmethod
    def _resize_pad(img, bboxes, input_size_w, input_size_h):
        iw, ih = img.size

        scale = min(input_size_w / iw, input_size_h / ih)

        sh, sw = int(ih * scale), int(iw * scale)

        img = func.resize(img, size=(sh, sw))
        img = func.pad(img, padding=(0, 0, input_size_w - sw, input_size_h - sh), fill=(128, 128, 128))

        bboxes *= scale

        return img, bboxes

    @staticmethod
    def _hflip(img, bboxes):
        img = func.hflip(img)
        w, _ = img.size
        bboxes = bboxes[:, [2, 1, 0, 3]] * torch.tensor([-1., 1., -1., 1.]) + torch.tensor([w, 0., w, 0.])  # (x_min, y_min, x_max, y_max) -> (-x_max + w, y_min, -x_min + w, y_max)

        return img, bboxes

    def _pad_with_no_object(self, bboxes, labels):
        _bboxes = torch.zeros((self.max_len, 4))
        _labels = torch.zeros(self.max_len)

        if bboxes.shape[0] != 0 and labels.shape[0] != 0:
            _bboxes[:bboxes.shape[0], :] = bboxes
            _labels[:labels.shape[0]] = labels

        return _bboxes, _labels
