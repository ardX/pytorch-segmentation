from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class COKYDataset(BaseDataSet):
    """
    ADE20K dataset 
    http://groups.csail.mit.edu/vision/datasets/ADE20K/
    """
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.COKY_palette
        super(COKYDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["training", "validation"]:
            self.image_dir = os.path.join(self.root, 'images', self.split)
            self.label_dir = os.path.join(self.root, 'annotations', self.split)
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.png')]
        else: raise ValueError(f"Invalid split name {self.split}")
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32) - 1
        return image, label, image_id

class COKY(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.5, 0.5, 0.5]
        self.STD = [0.2, 0.2, 0.2]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = COKYDataset(**kwargs)
        super(COKY, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
