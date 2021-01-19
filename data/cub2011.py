import os
import pandas as pd
import numpy as np
import cv2
import random
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


class CUB(Dataset):
    def __init__(self, files_path, labels, train_test, image_name, train=True, 
                 transform=False):
      
        self.files_path = files_path
        self.labels = labels
        self.transform = transform
        self.train_test = train_test
        self.image_name = image_name
        
        if train:
          mask = self.train_test.is_train.values == 1
          
        else:
          mask = self.train_test.is_train.values == 0
        
        
        self.filenames = self.image_name.iloc[mask]
        self.labels = self.labels[mask]
        self.num_files = self.labels.shape[0]
       
    def read_image(self, path):
        im = cv2.imread(str(path))
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def center_crop(self, im, min_sz=None):
        """ Returns a center crop of an image"""
        r,c,*_ = im.shape
        if min_sz is None: min_sz = min(r,c)
        start_r = math.ceil((r-min_sz)/2)
        start_c = math.ceil((c-min_sz)/2)
        return self.crop(im, start_r, start_c, min_sz, min_sz)

    def crop(self, im, r, c, target_r, target_c): 
        return im[r:r+target_r, c:c+target_c]

    def random_crop(self, x, target_r, target_c):
        """ Returns a random crop"""
        r,c,*_ = x.shape
        rand_r = random.uniform(0, 1)
        rand_c = random.uniform(0, 1)
        start_r = np.floor(rand_r*(r - target_r)).astype(int)
        start_c = np.floor(rand_c*(c - target_c)).astype(int)
        return self.crop(x, start_r, start_c, target_r, target_c)

    def rotate_cv(self, im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
        """ Rotates an image by deg degrees"""
        r,c,*_ = im.shape
        M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
        return cv2.warpAffine(im,M,(c,r), borderMode=mode, 
                            flags=cv2.WARP_FILL_OUTLIERS+interpolation)

    def normalize(self, im):
        """Normalizes images with Imagenet stats."""
        imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return (im/255.0 - imagenet_stats[0])/imagenet_stats[1]

    def apply_transforms(self, x, sz=(224, 224), zoom=1.05):
        """ Applies a random crop, rotation"""
        sz1 = int(zoom*sz[0])
        sz2 = int(zoom*sz[1])
        x = cv2.resize(x, (sz1, sz2))
        x = self.rotate_cv(x, np.random.uniform(-10,10))
        x = self.random_crop(x, sz[1], sz[0])
        if np.random.rand() >= .5:
                    x = np.fliplr(x).copy()
        return x

    def denormalize(self, img):
        imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return img*imagenet_stats[1] + imagenet_stats[0]
        
    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        y = self.labels.iloc[index,1] - 1
        
        file_name = self.filenames.iloc[index, 1]
        path = self.files_path/'images'/file_name
        x = self.read_image(path)
        if self.transform:
            x = self.apply_transforms(x)
        else:
            x = cv2.resize(x, (224,224))
        x = self.normalize(x)
        x =  np.rollaxis(x, 2) # To meet torch's input specification(c*H*W) 
        return x,y

    