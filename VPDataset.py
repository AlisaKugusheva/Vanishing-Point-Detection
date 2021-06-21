import glob
import pandas as pd
import os
import numpy as np
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import Sequential
import cv2
import random
import json
from pathlib import Path
from tqdm.notebook import tqdm
import torchvision

class VPDataset(Dataset):

    def __init__(self, df, root_dir, transform=None, test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform  
        self.test = test
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        image = io.imread(self.root_dir+img_name).astype('double')
        coords = np.array([self.df.iloc[idx, 1:]]).astype('double')
        sample = image, coords
        if self.transform:
            sample = self.transform(sample)
        if self.test == True:
            return sample, img_name
        else:
            return sample

class ToTensor(object):
    def __call__(self, sample):
        image, coords = sample
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image.copy()).float(), torch.from_numpy(coords).float()
