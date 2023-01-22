import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import *
from others import *
from CelebA import *
import math 
import shutil
from numpy.random import default_rng
from copy import deepcopy
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from pytorch_grad_cam import XGradCAM




class CelebATrain:
    def __init__(self, device):
        self.model = ResNet50(2)
        self.optimizer    = optim.SGD(
                self.model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4
            )

        # Some settings needed for handling dataset images
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        target_resolution = (224, 224)
        self.transform_test = transforms.Compose([
                    transforms.CenterCrop(orig_min_dim),
                    transforms.Resize(target_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Datasets
        self.train_dataset = CelebADataset(
                # raw_data_path=self.args.dataset_dir,
                root='data/CelebA',
                split="train",
                transform=self.transform_train,
                download=True
            )
        self.val_dataset = CelebADataset(
                root='data/CelebA',
                split="valid",                     
                transform = self.transform_train, 
                download = True,            
            )
        self.test_dataset = CelebADataset(
                root='data/CelebA',
                split="test",                     
                transform = self.transform_test, 
                download = True,            
            )

        # Dataloaders
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=512,
                shuffle=True,
                num_workers=1
            )

        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=512,
                shuffle=False,
                num_workers=1
            )
        self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=512,
                shuffle=False,
                num_workers=1
            )

       

















        

            

