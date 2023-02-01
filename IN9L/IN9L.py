from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from tqdm import tqdm

import os

class IN9L_dataset(Dataset):
    def __init__(
            self,
            root,
            split,
            transform=None,
    ) -> None:
        super().__init__()
        self.split = split
        self.data_path = []
        self.targets = []
        self.transform = transform
        if split == 'train' or split == 'val':
            self.raw_img_data_dir = os.path.join(root, split)
        else:
            self.raw_img_data_dir = os.path.join(
                root, split, 'val')

        self.data_path = []
        self.targets = []
        data_class_names = sorted(os.listdir(self.raw_img_data_dir))
        print("-"*10, f"indexing {self.split} data", "-"*10)
        for data_class_name in tqdm(data_class_names):
            try:
                target = int(data_class_name.split('_')[0])
            except:
                continue
            class_image_file_paths = glob(
                os.path.join(self.raw_img_data_dir, data_class_name, '*'))
            self.data_path += class_image_file_paths
            self.targets += [target] * len(class_image_file_paths)

    def __len__(self):
        return len(self.data_path)

   
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, img_file_path, target) where target is index of the target class.
        """
        target = self.targets[index]
        img = Image.open(self.data_path[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.data_path[index], target