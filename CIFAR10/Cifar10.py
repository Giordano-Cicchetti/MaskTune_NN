from torchvision.datasets import CIFAR10
from PIL import Image
from tqdm import tqdm
from glob import glob

import os
import shutil

import pandas as pd
import numpy as np



class Cifar10(CIFAR10):
    def __init__(
            self,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        #self.split = kwargs["train"]


        if (self.train):
            dataset="train"
            self.img_data_dir = os.path.join(
                kwargs["root"],
                "train")
        else:
            dataset="test"
            self.img_data_dir = os.path.join(
                kwargs["root"],
                "test")
        
        

        if not (os.path.isdir(self.img_data_dir) and len(os.listdir(self.img_data_dir)) > 0):
            
            print(
                f"\n\nstart creating and saving {dataset} dataset of Cifar10\n\n")
            
            os.makedirs(self.img_data_dir,exist_ok=True)

            for label in np.unique(self.targets):
                os.makedirs(os.path.join(
                    self.img_data_dir, str(label)), exist_ok=True)

            #tqdm(zip(image_ids, labels, confounders, partitions), total=len(image_ids)):

            for image_id, (image, label) in enumerate(zip(self.data, self.targets)):
                Image.fromarray(image).save(os.path.join(
                    self.img_data_dir, str(label), f'{image_id}.png'))
                
            print(f"\n\nfinished creating and saving {dataset} dataset of CelebA\n\n")
            
        print(
                f"\n\nstart loading in main memory {dataset} dataset of Cifar10\n\n")

        self.data_path = []
        self.targets = []
        data_classes = sorted(os.listdir(self.img_data_dir))
        for img_class in tqdm(data_classes):
            label=int(img_class)
            image_file_paths = glob(
                os.path.join(self.img_data_dir, img_class, '*'))
            self.data_path += image_file_paths
            self.targets += [label]*len(image_file_paths)


    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index: int):

        # Using id, you take image and label related to that value
        img_file_path, label = self.data_path[index], self.labels[index]

        # Obtain image from the path
        img = Image.open(img_file_path)
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, img_file_path, label

        