import numpy as np
import torch
import torchvision.transforms.functional as cv_func

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


def resize_img(img, rw, rh):
    iw, ih = img.size
    resize_scale = min(rw / iw, rh / ih)
    sh, sw = int(ih * resize_scale), int(iw * resize_scale)

    img = cv_func.resize(img, size=(sh, sw))

    return img, resize_scale


def pad_img(img, rw, rh):
    iw, ih = img.size

    pad_width = (rw - iw) / 2
    pad_height = (rh - ih) / 2

    pad_left, pad_right = int(np.floor(pad_width)), int(np.ceil(pad_width))
    pad_top, pad_bottom = int(np.floor(pad_height)), int(np.ceil(pad_height))

    img = cv_func.pad(img, padding=(pad_left, pad_top, pad_right, pad_bottom))
    pad_mask = torch.ones((rh, rw)).bool()
    pad_mask[pad_top:(rh - pad_bottom), pad_left:(rw - pad_right)] = False

    return img, pad_mask


def norm_img(img):
    img = cv_func.to_tensor(img)
    img = cv_func.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img
