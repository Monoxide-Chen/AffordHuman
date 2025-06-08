import torch
import os
import random
import numpy as np
import yaml
import torch.distributed as dist
from tools.trainer import train
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataset_utils.dataset_3DIR import _3DIR
from lemon_3d.tools.models.model_LEMON_d_laso import LEMON

img_file = 'Data/txt_scripts/train.txt'
obj_file = 'Data/txt_scripts/Point_train.txt'
human_file = 'Data/smplh_param/human_label.json'
behave_file = 'Data/Behave/behave.json'
dataset = _3DIR(img_file, obj_file, human_file, behave_file, mode='train')
data = dataset[0]
for key in data.keys():
    print(key)
print(len(dataset))