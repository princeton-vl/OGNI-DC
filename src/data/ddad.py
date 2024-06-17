import os
from glob import glob
import os.path as osp

import numpy as np
import json
import random
from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

"""
Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
            https://github.com/bartn8/vppdc/blob/main/dataloaders/datasets.py
"""

def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

class DDAD(BaseDataset):
    def __init__(self, args, mode):
        super(DDAD, self).__init__(args, mode)

        self.args = args
        assert mode=='test'
        self.mode = mode

        self.sample_list = []
        self.extra_info = []
        self.calib_list = []

        datapath = self.args.dir_data

        image_list = sorted(glob(osp.join(datapath, 'rgb/*.png')))
        gt_list = sorted(glob(osp.join(datapath, '*gt/*.png')))
        hints_list = sorted(glob(osp.join(datapath, 'hints/*.png')))
        calibtxt_list = sorted(glob(osp.join(datapath, 'intrinsics/*.txt')))

        # Filter data not present in other folders
        for i in range(len(image_list)):
            self.sample_list += [[image_list[i], gt_list[i], hints_list[i]]]
            self.extra_info += [[image_list[i].split('/')[-1], False]]  # scene and frame_id and do flip
            self.calib_list += [[calibtxt_list[i]]]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        K = np.loadtxt(self.calib_list[idx][0])
        rgb = Image.open(self.sample_list[idx][0])
        gt = read_depth(self.sample_list[idx][1])
        depth = read_depth(self.sample_list[idx][2])

        depth = Image.fromarray(depth.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')



        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225), inplace=True)

        depth = TF.to_tensor(np.array(depth))

        gt = TF.to_tensor(np.array(gt))

        # print('dataloader', torch.max(gt[gt>0.0]), torch.min(gt[gt>0.0]))

        output = {'rgb': rgb, 'dep': depth, 'gt': gt, 'K': torch.Tensor(K)}

        return output