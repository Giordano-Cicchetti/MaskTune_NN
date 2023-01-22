from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from glob import glob

import os
import shutil

import pandas as pd
import numpy as np


data_split = {
    0: 'train',
    1: 'val',
    2: 'test'
}

class CelebADataset(CelebA):
    def __init__(
            self,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.split = kwargs["split"]

        self.target_attribute = "Blond_Hair"
        self.transform = kwargs["transform"]
        self.confounder =  "Male"
        self.data_path = {}
        self.labels = {}
        self.confounders = {}
        self.split = kwargs["split"]
        ### Da Definire ###
        self.raw_data_path = ''
        ######
        self.root = kwargs["root"]

        # Give a name to the dataset's folder
        if (self.split=="train"):
            self.img_data_dir = os.path.join(
                kwargs["root"],
                self.split)
        elif (self.split=="test"):
            self.img_data_dir = os.path.join(
                kwargs["root"],
                self.split)
        else:
            self.img_data_dir = os.path.join(
                kwargs["root"],
                self.split)
        
        
        if not (os.path.isdir(self.img_data_dir) and len(os.listdir(self.img_data_dir)) > 0):
            print(
                f"\n\nstart creating and saving {self.split} dataset of CelebA\n\n")
            attrs_df = pd.read_csv(os.path.join(
                self.raw_data_path, 'list_attr_celeba.csv'))

            # Salvo gli ids e li rimuovo dal dataset
            image_ids = attrs_df['image_id'].values
            attrs_df = attrs_df.drop(labels='image_id', axis='columns')

            # Prendo i nomi degli attributi
            attr_names = attrs_df.columns.copy()
            attrs_df = attrs_df.values
            
            attrs_df[attrs_df == -1] = 0

            # Setto il parametro target
            target_idx = attr_names.get_loc(self.target_attribute)
            labels = attrs_df[:, target_idx]

            # Prendo il parametro confounder
            confounder_idx = attr_names.get_loc(self.confounder)
            confounders = attrs_df[:, confounder_idx]

            # Partiziono gli elementi del dataset a seconda di quale set faranno parte {train, test, validation}
            partition_df = pd.read_csv(os.path.join(
                self.raw_data_path, 'list_eval_partition.csv'))
            partitions = partition_df['partition']

            # Creo la cartella relativa al dataset
            os.makedirs(self.img_data_dir, exist_ok=True)


            for image_id, label, confounder, partition in tqdm(zip(image_ids, labels, confounders, partitions), total=len(image_ids)):
                if data_split[partition] == self.split:
                  shutil.copy(os.path.join('img_align_celeba', 'img_align_celeba', image_id), os.path.join(
                      self.img_data_dir, image_id))
                  self.data_path[int(image_id.split('.')[-2])] = os.path.join(self.img_data_dir, image_id)
                  self.labels[int(image_id.split('.')[-2])] = int(label)
                  self.confounders[int(image_id.split('.')[-2])] = confounder
                
            print(f"\n\nfinished creating and saving {self.split} dataset of CelebA\n\n")
            return




        attrs_df = pd.read_csv(os.path.join(
                self.raw_data_path, 'list_attr_celeba.csv'))

        # Salvo gli ids e li rimuovo dal dataset
        image_ids = attrs_df['image_id'].values
        attrs_df = attrs_df.drop(labels='image_id', axis='columns')

        # Prendo i nomi degli attributi
        attr_names = attrs_df.columns.copy()
        attrs_df = attrs_df.values
            
        attrs_df[attrs_df == -1] = 0

        # Setto il parametro target
        target_idx = attr_names.get_loc(self.target_attribute)
        labels = attrs_df[:, target_idx]

        # Prendo il parametro confounder
        confounder_idx = attr_names.get_loc(self.confounder)
        confounders = attrs_df[:, confounder_idx]

        # Partiziono gli elementi del dataset a seconda di quale set faranno parte {train, test, validation}
        partition_df = pd.read_csv(os.path.join(
          self.raw_data_path, 'list_eval_partition.csv'))
        partitions = partition_df['partition']

        for image_id, label, confounder, partition in tqdm(zip(image_ids, labels, confounders, partitions), total=len(image_ids)):
          if data_split[partition] == self.split:
            self.data_path[int(image_id.split('.')[-2])] = os.path.join(self.img_data_dir, image_id)
            self.labels[int(image_id.split('.')[-2])] = int(label)
            self.confounders[int(image_id.split('.')[-2])] = confounder

                 
        print(f"\n\nfinished creating and saving {self.split} dataset of CelebA\n\n")  

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index: int):

      # Using id, you take image and label related to that value
      img_file_path, label = self.data_path[index], self.labels[index]

      # Obtain image from the path
      img = Image.open(img_file_path)
          
      if self.transform is not None:
        img = self.transform(img)
          
      confounder = self.confounders[index]
          
      return img, img_file_path, label, confounder