from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from tqdm import tqdm
import os

#SubClass of Dataset that takes the IN9L dataset stored in the folder
#indicated by the parameter "root" and perform operation on it
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
        #Determine the raw_img data directory based on the dataset we want to create
        if split == 'train' or split == 'val':
            self.raw_img_data_dir = os.path.join(root, split)
        else:
            self.raw_img_data_dir = os.path.join(
                root, split, 'val')
        #Create the variables data_path and targets
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
        # Using index, you take image and label related to that value
        target = self.targets[index]
        path = self.data_path[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        #Return (img, data_path, target)
        return img, path , target