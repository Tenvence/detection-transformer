import os

import PIL.Image as Image
import torch
import torchvision.datasets

from .coco_tools import resize_img, pad_img, norm_img


class CocoObjectDetectionTest(torchvision.datasets.CocoDetection):
    def __init__(self, root, ann_file, input_size):
        super().__init__(root, ann_file)
        self.input_size_h, self.input_size_w = input_size

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])).convert('RGB')
        iw, ih = img.size
        img, resize_scale = resize_img(img, self.input_size_w, self.input_size_h)
        img, pad_mask = pad_img(img, self.input_size_w, self.input_size_h)
        img = norm_img(img)
        return img, pad_mask, img_id, iw, ih, resize_scale
