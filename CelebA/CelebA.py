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
    1: 'valid',
    2: 'test'
}

#SubClass of Dataset that takes the CelebA dataset stored in the folder
#indicated by the parameter "root" and perform operation on it
class CelebADataset(Dataset):
    def __init__(
            self,
            root,
            split,
            transform=None
    ) -> None:
        super().__init__()
        self.split = split
        self.target_attribute = "Blond_Hair"
        self.transform = transform
        self.confounder =  "Male"
        self.root = root
        self.data_path = []
        self.labels = []
        #In confounder we store de information about the genre of each image 
        self.confounders = {}

        if (self.split=="train"):
            self.img_data_dir = os.path.join(
                self.root,
                self.split)
        elif (self.split=="test"):
            self.img_data_dir = os.path.join(
                self.root,
                self.split)
        else:
            self.img_data_dir = os.path.join(
                self.root,
                self.split)
        
        #If it is the first time that we download the dataset:
        if not (os.path.isdir(self.img_data_dir) and len(os.listdir(self.img_data_dir)) > 0):
            print(f"\n\nstart creating and saving {self.split} dataset of CelebA\n\n")
            attrs_df = pd.read_csv(os.path.join(
                self.root, 'list_attr_celeba.csv'))

            # Save ids e remove them from dataset
            image_ids = attrs_df['image_id'].values
            attrs_df = attrs_df.drop(labels='image_id', axis='columns')

            # Take columns' name
            attr_names = attrs_df.columns.copy()
            attrs_df = attrs_df.values
            attrs_df[attrs_df == -1] = 0

            # Set target attribute
            target_idx = attr_names.get_loc(self.target_attribute)
            labels = attrs_df[:, target_idx]

            # take confounder values
            confounder_idx = attr_names.get_loc(self.confounder)
            confounders_val = attrs_df[:, confounder_idx]

            # Take element of the actual partition: {train, test, validation}
            partition_df = pd.read_csv(os.path.join(
                self.root, 'list_eval_partition.csv'))
            partitions = partition_df['partition']

            # Make two folders: "0" and "1"
            for label in np.unique(labels):
                os.makedirs(os.path.join(
                    self.img_data_dir, str(label)), exist_ok=True)

            print("-"*10, f"indexing {self.split} data", "-"*10)
            for image_id, label, confounder, partition in tqdm(zip(image_ids, labels, confounders_val, partitions), total=len(image_ids)):
                if data_split[partition] == self.split:
                  # Create training Dataset
                  shutil.copy(os.path.join(self.root,'img_align_celeba', 'img_align_celeba', image_id), os.path.join(
                      self.img_data_dir, str(label),image_id))
                  self.data_path.append(os.path.join(self.img_data_dir, str(label),image_id))
                  self.labels.append(int(label))
                  self.confounders[image_id] = confounder

            print(f"\n\nfinished creating and saving {self.split} dataset of CelebA\n\n")
            return

        print("-"*10, f"indexing {self.split} data", "-"*10)
        attrs_df = pd.read_csv(os.path.join(
                self.root, 'list_attr_celeba.csv'))

        # Save ids e remove them from dataset
        image_ids = attrs_df['image_id'].values
        attrs_df = attrs_df.drop(labels='image_id', axis='columns')

        # Take columns' name
        attr_names = attrs_df.columns.copy()
        attrs_df = attrs_df.values
            
        attrs_df[attrs_df == -1] = 0

        # Set target attribute
        target_idx = attr_names.get_loc(self.target_attribute)
        labels = attrs_df[:, target_idx]

        # take confounder values
        confounder_idx = attr_names.get_loc(self.confounder)
        confounders_val = attrs_df[:, confounder_idx]

        # Take element of the actual partition: {train, test, validation}
        partition_df = pd.read_csv(os.path.join(
          self.root, 'list_eval_partition.csv'))
        partitions = partition_df['partition']

        for image_id, confounder, partition in tqdm(zip(image_ids, confounders_val, partitions), total=len(image_ids)):
          if data_split[partition] == self.split:
            self.confounders[image_id] = confounder
        
        self.data_path = []
        self.labels    = []

        data_classes = sorted(os.listdir(self.img_data_dir))
        for data_class in tqdm(data_classes):
            try:
                label = int(data_class)
            except:
                continue
            class_image_file_paths = glob(
                os.path.join(self.img_data_dir, data_class, '*'))
            self.data_path += class_image_file_paths
            
            self.labels += [label] * len(class_image_file_paths)


                 
        print(f"\n\nfinished creating and saving {self.split} dataset of CelebA\n\n")    

    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index: int):

        # Using index, you take image and label related to that value
        img_file_path, label = self.data_path[index], self.labels[index]
        # Using the last part of file_path to access confounder dict
        confounder = self.confounders[img_file_path.split('/')[-1]]
        # Obtain image from the path
        img = Image.open(img_file_path)
            
        if self.transform is not None:
            img = self.transform(img)
        
            
        return img, img_file_path, label, confounder

        