import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from model import *
from others import *
from IN9L import IN9L_dataset
import math 
import shutil
from numpy.random import default_rng
from copy import deepcopy
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from pytorch_grad_cam import XGradCAM
import numpy as np
from torchvision.utils import save_image
from glob import glob
import cv2
from PIL import Image

class BackgroundChallenge_Train:
    def __init__(self, device):
        #DEVICE
        self.device       = device
        #LOSS FUNCTION
        self.loss_function= nn.CrossEntropyLoss()
        #MODEL
        self.model = ResNet50(num_classes=9,pretrained=True)
        self.model.to(device)
        #OPTIMIZER
        self.optimizer    = optim.SGD(
                self.model.parameters(),
                lr=0.0001,
                momentum=0.9,
                weight_decay=1e-4
            )
        #SCHEDULER
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=[i for i in range(30,101,30)],
                gamma= 0.1,
                last_epoch=-1,
            )
        #LOGGER
        self.logger       = Logger("", None)
    
        # Some default transformations 
        # Standard normalization for pretrained networks on ImageNet
        self.transform_test = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(
                    (224,224),
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.initialize_datasets_loaders()
        
    def initialize_datasets_loaders(self):
        # Creation Train and Validation Datasets
        self.train_dataset = IN9L_dataset(
            root="data/original",
            split='train',
            transform=self.transform_train
        )
                
        self.val_dataset = IN9L_dataset(
            root="data/original",
            split='val',
            transform=self.transform_test
        )

        # Creation Test Datasets
        self.test_dataset_orig = IN9L_dataset(
            root="data/bg_challenge",
            split='original',
            transform=self.transform_test
        )
        
        self.test_dataset_ms = IN9L_dataset(
            root="data/bg_challenge",
            split='mixed_same',
            transform=self.transform_test
        )
        
        self.test_dataset_mr = IN9L_dataset(
            root="data/bg_challenge",
            split='mixed_rand',
            transform=self.transform_test
        )
        
        self.test_dataset_fg = IN9L_dataset(
            root="data/bg_challenge",
            split='only_fg',
            transform=self.transform_test
        )
        

        # Creation loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=128, shuffle=True, num_workers=1)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=128, shuffle=False, num_workers=1)

        self.test_loader_orig = torch.utils.data.DataLoader(
            self.test_dataset_orig, batch_size=128, shuffle=False, num_workers=1)
        
        self.test_loader_ms = torch.utils.data.DataLoader(
            self.test_dataset_ms, batch_size=128, shuffle=False, num_workers=1)
        
        self.test_loader_mr = torch.utils.data.DataLoader(
            self.test_dataset_mr, batch_size=128, shuffle=False, num_workers=1)

        self.test_loader_fg = torch.utils.data.DataLoader(
            self.test_dataset_fg, batch_size=128, shuffle=False, num_workers=1)

        self.test_loaders = [self.test_loader_orig, self.test_loader_ms, self.test_loader_mr, self.test_loader_fg]
        self.test_loaders_names = ["loader original", "loader mixed same", "loader mixed rand", "loader only foreground"]

    def mask_data(self, checkpoint_path=None):
        #Load the trained model for masking the image
        if(checkpoint_path!=None):
          checkpoint = torch.load(checkpoint_path)
          self.model.load_state_dict(checkpoint['model_state_dict'])
          self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          
        #Use XGradCAM as mask generator 
        heat_map_generator = XGradCAM(
            model=self.model,
            target_layers=[self.model.get_grad_cam_target_layer()],
            use_cuda=True,
        )
        
        transform_data_to_mask = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        #Generate masked dataset starting from train dataset
        self.masked_dataset = deepcopy(self.train_dataset)
        self.masked_dataset.transform=transform_data_to_mask
        masked_data_dir = self.train_dataset.raw_img_data_dir.replace("train", "masked")
        if not (os.path.isdir(masked_data_dir) and len(os.listdir(masked_data_dir)) > 0):
            os.makedirs(masked_data_dir, exist_ok=True)
            os.mkdir(os.path.join(masked_data_dir,"00_dog"))
            os.mkdir(os.path.join(masked_data_dir,"01_bird"))
            os.mkdir(os.path.join(masked_data_dir,"02_wheeled vehicle"))
            os.mkdir(os.path.join(masked_data_dir,"03_reptile"))
            os.mkdir(os.path.join(masked_data_dir,"04_carnivore"))
            os.mkdir(os.path.join(masked_data_dir,"05_insect"))
            os.mkdir(os.path.join(masked_data_dir,"06_musical instrument"))
            os.mkdir(os.path.join(masked_data_dir,"07_primate"))
            os.mkdir(os.path.join(masked_data_dir,"08_fish"))

            
            for data in tqdm(self.train_loader):
                # Creation Heat-Map for each batch
                images,data_paths,labels = data[0],data[1],data[2]
                
                hm = heat_map_generator(images)
        
                # Mask Creation
                mask_mean_value = np.nanmean(np.where(hm > 0, hm, np.nan), axis=(1, 2))[:, None, None]
                mask_std_value = np.nanstd(np.where(hm > 0, hm, np.nan), axis=(1, 2))[:, None, None]
                mask_threshold_value = mask_mean_value + 2 * mask_std_value
                masks = np.where(hm > mask_threshold_value, 0, 1)
                

                # Mask application on images
                for image,mask,original_path in zip(images,masks,data_paths):
                    
                    original_image = Image.open(original_path).convert('RGB')
                    image_mask = np.expand_dims(cv2.resize(mask, dsize=original_image.size, interpolation=cv2.INTER_NEAREST), axis=-1)
                    masked_image = np.array(original_image) * image_mask
                    path=original_path.replace("train", "masked")
                    im = Image.fromarray(masked_image.astype(np.uint8))
                    im.save(path)
        
        self.masked_dataset.data_path = []
        self.masked_dataset.targets    =[]
        
        data_class_names = sorted(os.listdir(masked_data_dir))
        print("-"*10, f"indexing Masked data", "-"*10)
        for data_class_name in tqdm(data_class_names):
            try:
                target = int(data_class_name.split('_')[0])
            except:
                continue
            class_image_file_paths = glob(
                os.path.join(masked_data_dir, data_class_name, '*'))
            self.masked_dataset.data_path += class_image_file_paths
            self.masked_dataset.targets += [target] * len(class_image_file_paths)
        
        
        self.masked_loader= torch.utils.data.DataLoader(
            self.masked_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=1
        )


    #Function used to run an epoch (train, validation or test)
    def run_an_epoch(self, data_loader, epoch, mode="train", device='cpu'):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        #variables used to calculate mean accuracy and mean loss in an epoch
        losses = AverageMeter()
        accuracies = AverageMeter()
        with torch.set_grad_enabled(mode == "train"):
            progress_bar = tqdm(data_loader)
            self.logger.info(
                f"{mode} epoch: {epoch}"
            )
            for data in progress_bar:
                progress_bar.set_description(f'{mode} epoch {epoch}')
                #Take batch of inputs in variable X and corrispondent labels in variable y
                X, y = data[0], data[2]
                X, y = X.to(device), y.to(device)
                #Inference time
                outputs = self.model(X)
                y = y.to(torch.int64)
                #Loss calculation 
                loss = self.loss_function(outputs, y)
                losses.update(loss.item(), X.size(0))
                #Use softmax and then argmax to assign integer label in [0,1] to outputs
                output_probabilities = F.softmax(outputs, dim=1)
                probabilities, y_pred = output_probabilities.data.max(1)
                #Calculate and update mean accuracy based on y and y_pred
                accuracies.update(calculate_accuracy(y, y_pred), 1)
                #Backward propagation of the gradient
                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                progress_bar.set_postfix(
                    {
                        "loss": losses.avg,
                        "accuracy": accuracies.avg,
                    }
                )
            self.logger.info(
                f"loss: {losses.avg}"
            )
            self.logger.info(
                f"accuracy: {accuracies.avg}"
            )
        return accuracies.avg


    #Function used to train the entire ResNet50 network for a certan number of epochs
    def train_erm(self,epochs=100, best_resume_checkpoint_path: str=None, last_resume_checkpoint_path: str=None) -> None:
        resume_epoch = 0
        self.best_accuracy=-math.inf
        if(best_resume_checkpoint_path!=None and last_resume_checkpoint_path!=None):
            checkpoint = torch.load(last_resume_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            resume_epoch = checkpoint['epoch'] + 1

            checkpoint = torch.load(best_resume_checkpoint_path)
            self.best_accuracy=checkpoint['accuracy']
        

        for current_epoch in range(resume_epoch,epochs):
            self.current_epoch = current_epoch
            #Run a train epoch
            self.run_an_epoch(
                data_loader=self.train_loader, epoch=current_epoch, mode="train",device=self.device)
            #Run a validation epoch and save the validation accuracy
            val_accuracy = self.run_an_epoch(
                data_loader=self.val_loader, epoch=current_epoch, mode="validation",device=self.device
            )
            self.lr_scheduler.step()

            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'accuracy' : val_accuracy,
                }, "last_erm_model.pt")
            if(val_accuracy>self.best_accuracy):
                self.best_accuracy=val_accuracy
                torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'accuracy' : val_accuracy,
                }, "best_erm_model.pt")


    #function used to test the accuracy of the model against the CelebA dataset
    def test(self, checkpoint_path=None):
        self.logger.info("-" * 10 + "testing the model" +"-" * 10, print_msg=True)
        #LOAD THE MODEL SPECIFIED IN THE CHECKPOINT PATH
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']

        #RUN AN EPOCH IN TEST MODE AND USING TEST LOADERS
        for id, test_loader in zip(self.test_loaders_names, self.test_loaders):
            self.logger.info("-" * 10 + f"Testing {id} accuracy" +"-" * 10, print_msg=True)
            accuracy = self.run_an_epoch(
                data_loader=test_loader, epoch=epoch,device=self.device,mode='test'
            )
            
    
   
    