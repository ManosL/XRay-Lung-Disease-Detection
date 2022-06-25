from math import floor
import os
import pandas as pd
import numpy as np

import cv2

import torch
from torch.utils.data import Dataset

class xRaysDataset(Dataset):
    # Change these percentages if needed.
    # If train=True I'll take the first 90% images of each label.
    # If train=False I'll take the last 10% images of each label.
    def __init__(self, dataset_dir, train=True, to_mask=False, transform=None, target_transform=None):
        TRAIN_SPLIT     = 0.9
        distinct_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

        images = []
        labels = []

        # Note: Images have size 299x299 pixels instead of 256x256
        for label in distinct_labels:
            meta_file_name = label + '.metadata.csv'

            meta_file_path = os.path.join(dataset_dir, meta_file_name)

            curr_df = pd.read_csv(meta_file_path, sep='|')

            curr_images = list(curr_df['FILE NAME'])

            split_index = floor(len(curr_images) * TRAIN_SPLIT)

            if train == True:
                curr_images = curr_images[0:split_index]
            else:
                curr_images = curr_images[split_index:]
            
            if label == 'Normal':
                curr_images = [img.title() for img in curr_images]

            images = images + curr_images
            labels = labels + ([distinct_labels.index(label)] * len(curr_images))

        self.dataset_dir      = dataset_dir
        self.images           = images

        self.distinct_labels  = distinct_labels
        self.labels           = labels
        
        self.to_mask          = to_mask
        self.transform        = transform
        self.target_transform = target_transform

        return



    def __len__(self):
        return len(self.labels)



    def __read_greyscale_image(self, img_path):
        assert(os.path.exists(img_path))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # I do it for now because it will be done either way and
        # also its useful to apply the mask.
        return image / 255



    def __process_mask(self, mask_image):
        pixels_to_add = floor((299 - 256) / 2)

        new_mask = cv2.copyMakeBorder(mask_image, pixels_to_add, pixels_to_add+1, 
                                    pixels_to_add, pixels_to_add+1, cv2.BORDER_CONSTANT,
                                    value=[0,0,0])

        return new_mask



    def __getitem__(self, idx):
        img_name   = self.images[idx]
        label      = self.labels[idx]
        label_name = self.distinct_labels[label]

        img_path  = os.path.join(self.dataset_dir, label_name, 'images', img_name + '.png')
        mask_path = os.path.join(self.dataset_dir, label_name, 'masks',  img_name + '.png')

        assert(os.path.exists(img_path))
        assert(os.path.exists(mask_path))

        img = self.__read_greyscale_image(img_path)

        if self.to_mask == True:
            mask = self.__read_greyscale_image(mask_path)
            mask = self.__process_mask(mask)

            img = np.multiply(img, mask)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)
        
        img = img.type(torch.FloatTensor)

        return img, label

"""
import matplotlib.pyplot as plt

DATASET_PATH = '/home/manosl/Desktop/MSc Courses Projects/2nd Semester/Deep Learning/Project 1/data/COVID-19_Radiography_Dataset'
ds = xRaysDataset(DATASET_PATH, train=True, to_mask=True)

print(len(ds))

i, l = ds[10]

plt.imshow(i)
plt.show()
"""