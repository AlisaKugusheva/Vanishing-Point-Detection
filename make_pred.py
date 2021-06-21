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
import argparse

from VPDataset import VPDataset, ToTensor
from architecture import Mobilenetv2


def predict(markup, test_dir, model_path):

    df_test = pd.read_json(markup).T
    df_test.reset_index(level=0, inplace=True)
    df_test.columns = ['image', 'x_label', 'y_label']
    
    test_set = VPDataset(df=df_test,
                                root_dir=test_dir,
                                transform=transforms.Compose([
                                                   ToTensor()]),
                                test=True)
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    model = Mobilenetv2().to('cpu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    results = {}
    
    with torch.no_grad():
            for (image, label), img_name in tqdm(test_loader):
                results[img_name[0]] = model(image).tolist()[0]
    return results

def save_predictions(predicted_json_path, results):
    with open(predicted_json_path, 'w') as f:
        json.dump(results, f)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model path',
                        default='./model.pt')
    parser.add_argument(
        '--test', help='Path to the test set', default='./test/')
    parser.add_argument(
        '--pred', help='name of json file to save', default='./predicted.json')
    parser.add_argument(
        '--gt', help='name of json file with gt', default='.test/markup.json')

    args = parser.parse_args()

    results = predict(args.gt, args.test, args.model)
    save_predictions(args.pred, results)

if __name__ == "__main__":
    main()
