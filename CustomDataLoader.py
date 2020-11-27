
import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import global_config

import configparser


cfg = configparser.ConfigParser()
cfg.read(global_config.model_config)


class CustomDataset(Dataset):

    def __init__(self, img_dir: str):
        global cfg
        paths = os.listdir(img_dir)
        abspath = os.path.abspath(img_dir)
        self.img_dir = img_dir
        self.paths = [os.path.join(abspath, path) for path in paths]
        self.list_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean = (cfg['MEAN'].getfloat('M'),cfg['MEAN'].getfloat('M2'),cfg['MEAN'].getfloat('M3')),
                                                                        std = (cfg['STD'].getfloat('S'),cfg['STD'].getfloat('S1'),cfg['STD'].getfloat('S2')))])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        global cfg
        path = self.paths[idx]
        text = self.get_filename(path)
        img = Image.open(path).convert('RGB')
        img = img.resize((cfg['DATASET'].getint('image_width'), cfg['DATASET'].getint('image_height')))
        img = self.transform(img)
        return img, text

    def get_filename(self, path: str) -> str:
        return os.path.basename(path).split('.')[0].lower().strip()

    def transform(self, img) -> torch.Tensor:
        return self.list_transforms(img)
