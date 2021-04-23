import torch
from torch.utils import data
import torch.nn.functional as F
import os, sys, getopt
from PIL import Image
import numpy as np
import albumentations as A

#DATA AUGMENTATION
from torchvision import transforms
prob = 0.5
pipeline_transform = A.Compose([
    A.VerticalFlip(p=prob),
    A.HorizontalFlip(p=prob),
    A.RandomRotate90(p=prob),
    A.ElasticTransform(alpha=0.1,p=prob),
    A.HueSaturationValue(hue_shift_limit=(-9),sat_shift_limit=25,val_shift_limit=10,p=prob),
    ])

#DATA NORMALIZATION
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset_patches(data.Dataset):

    def __init__(self, list_IDs, labels, dataset):

        self.labels = labels
        self.list_IDs = list_IDs
        self.dataset = dataset
        
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = Image.open(ID)
        X = np.asarray(X)
        y = self.labels[index]
        #data augmentation
        if (self.dataset=='train'):
            X = pipeline_transform(image=X)['image']
            X = np.asarray(X)
        #data transformation
        input_tensor = preprocess(X)
                
        return input_tensor, np.asarray(y)


class Dataset_WSI(data.Dataset):

    def __init__(self, list_IDs):

        self.list_IDs = list_IDs
        
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        # Select sample
        ID = self.list_IDs[index]
        #print(ID)
        # Load data and get label
        X = Image.open(ID)
        X = np.asarray(X)

        #data transformation
        input_tensor = preprocess(X)
                
        return input_tensor, ID

class Dataset_TMA_core(data.Dataset):

    def __init__(self, list_IDs, labels):

        self.labels = labels
        self.list_IDs = list_IDs
        
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = Image.open(ID)
        X = np.asarray(X)
        y = self.labels[index]

        #data transformation
        input_tensor = preprocess(X)
                
        return input_tensor, np.asarray(y)

if __name__ == "__main__":
	pass