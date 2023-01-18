from torchvision.datasets import MNIST
import torch
import os
from PIL import Image
import numpy as np
from glob import glob

class BiasedMNIST(MNIST):
    def __init__(
      self,
      bias_conflicting_data_ratio=0.1,
      biased=False,
      **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train = kwargs["train"]
 
        self.biased=biased
        if kwargs["train"]:
            self.split = "train"
        else:
            self.split = "test"

        #to do if su argomento biased
        if(self.split=="train"):
            self.img_data_dir = os.path.join(
                kwargs["root"],
                "MNISTRGB_BIASED",
                self.split)

        elif(self.split=="test" and biased==False):
            self.img_data_dir = os.path.join(
                kwargs["root"],
                "MNISTRGB_BIASED",
                self.split,
                "original")

        elif(self.split=="test" and biased==True):
            self.img_data_dir = os.path.join(
                kwargs["root"],
                "MNISTRGB_BIASED",
                self.split,
                "biased")
        
        print(self.img_data_dir)
        self.data_new=[]
        self.data_new_paths=[]

        filtered_target_idx = torch.cat(
            [torch.where(self.targets == label)[0] for label in [0,1,2,3,4,5,6,7,8,9]])
        self.data,self.targets=self.data[filtered_target_idx], self.targets[filtered_target_idx]

        #modifico label: [0,1,2,3,4] -> 0  [5,6,7,8,9] -> 1
        self.targets = self.modify_labels()
        self.elem_in_class_zero = torch.count_nonzero(self.targets).item()
        self.elem_in_class_one = len(self.targets)-self.elem_in_class_zero
        
        
        #Se è la prima volta che creo questo dataset
        if not (os.path.isdir(self.img_data_dir) and len(os.listdir(self.img_data_dir)) > 0):
            print(
                f"\n\nstart creating and saving {self.split} dataset of BiasedMnist\n\n"
            )
            os.makedirs(self.img_data_dir, exist_ok=True)
            
            #data_new conterrà le immagini di mnist ma in rgb oppure in rgb e biased 
            if(self.split=="train"):
                self.data_new = self.add_bias_to_images(bias_ratio=0.01)
            elif(biased==False):
                self.data_new = self.from_BlackWhite_to_RGB()
            else:
                self.data_new = self.add_bias_to_images()

            #bisogna salvare le nuove immagini per prenderle poi in un secondo momento
            for id, (data, target) in enumerate(zip(self.data_new, self.targets)):
                Image.fromarray(data.numpy().astype(np.uint8)).save(
                    os.path.join(self.img_data_dir, f"{id}.png")
                )
                self.data_new_paths.append(os.path.join(self.img_data_dir,f"{id}.png"))
       
        #create variables data_new e data_new_paths
        self.data_new=[]
        self.data_new_paths=[]
        
        image_file_paths = sorted(glob(
            os.path.join(self.img_data_dir, "*")
        ))
        self.data_new_paths += image_file_paths
        for image_path in image_file_paths:
          temp = Image.open(image_path)
          keep = temp.copy()
          self.data_new.append(keep)
          temp.close()
              
    #modify labels according to this map: [0,1,2,3,4]->0 [5,6,7,8,9]->1
    def modify_labels(self):
      return torch.tensor(np.where(self.targets<5,0,1))

    #add two dimensions to original images in order to create RGB images
    def from_BlackWhite_to_RGB(self):
       data_to_transform = self.data.clone()
       data_to_transform = torch.unsqueeze(data_to_transform, dim=-1).repeat((1, 1, 1, 3))
       return data_to_transform
    
    def add_bias_to_images(self, bias_ratio=0.1 ,color=(255,0,0) , mask_width=4):
        data_to_transform = self.data.clone()
        data_to_transform = torch.unsqueeze(data_to_transform, dim=-1).repeat((1, 1, 1, 3))

        if (self.split=="train"):
            #99% of class0 biased and 1% of class1 biased
            to_bias_zero= round(self.elem_in_class_zero*(1-bias_ratio))
            to_bias_one = round(self.elem_in_class_one*bias_ratio)

            biased_zero = 0
            biased_one  = 0
            
            for index,elem in enumerate(data_to_transform):
                target=self.targets[index]
                if(target==0):
                    if(biased_zero>to_bias_zero): 
                        continue
                    for i,component in enumerate(color):
                        elem[:mask_width,:mask_width,i]=component
                    biased_zero += 1

                elif(target==1):
                    if(biased_one>to_bias_one): 
                        continue
                    for i,component in enumerate(color):
                        elem[:mask_width,:mask_width,i]=component
                    biased_one += 1
                    

        elif( self.split=="test"):
            #all class1 biased , all class0 original
            for index,elem in enumerate(data_to_transform):
                target=self.targets[index]
                if(target==1):
                    for i,component in enumerate(color):
                        elem[:mask_width,:mask_width,i]=component
            
        return data_to_transform


    def __len__(self):
        return len(self.data_new)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, img_file_path, target) where target is index of the target class.
        """
        img, img_file_path, target = self.data_new[index], self.data_new_paths[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, img_file_path, target

