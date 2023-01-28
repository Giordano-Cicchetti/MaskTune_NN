import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
from model import resnet32
from others import *
#from Cifar10 import Cifar10
from torchvision.datasets import CIFAR10
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
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler


class Cifar10Train:
    def __init__(self, device):
        #DEVICE
        self.device       = device
        #LOSS FUNCTION
        self.loss_function= nn.CrossEntropyLoss()
        #MODEL
        self.model = resnet32(num_classes=10)
        self.model.to(device)
        #OPTIMIZER
        self.optimizer    = optim.SGD(
                self.model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=1e-4
            )
        #LOGGER
        self.logger       = Logger("", None)
        #SCHEDULER 
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=[i for i in range(25,301,25)],
                gamma= 0.5,
                last_epoch=-1,
            )
    
        # Some settings needed for handling dataset images
        
        self.transform_test = transforms.Compose([transforms.ToTensor()])
        self.transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ]
        )
        transform_data_to_mask = transforms.Compose(
            [transforms.ToTensor(), ])
            
        # Datasets and Dataloaders
        self.initialize_datasets_loaders()

        
        
    def initialize_datasets_loaders(self):
        self.train_dataset = CIFAR10(
                root='data/',
                train=True,
                transform=self.transform_train,
                download=True
            )
        self.test_dataset = CIFAR10(
                root='data/',
                train=False,                     
                transform = self.transform_test, 
                download = True,            
            )

        self.val_dataset = CIFAR10(
                root='data/',
                train=True,                     
                transform = self.transform_test, 
                download = True,            
            )
        
        #Split Train dataset and build validation dataset
        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))
        np.random.seed(50)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)

        # Dataloaders
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=128,
                sampler=self.train_sampler,
                shuffle=False,
                num_workers=1
            )

        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=128,
                sampler=self.val_sampler,
                shuffle=False,
                num_workers=1
            )
        self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=1
            )


    def mask_data(self,checkpoint_path=None):
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
            transforms.ToTensor()
        ])

        #Generate masked dataset starting from train dataset
        self.masked_dataset = deepcopy(self.train_dataset)
        self.masked_dataset.transform=transform_data_to_mask

        masked_data=[]
        masked_labels=[]
        #maskedimg = torch.empty(1,3,224,224)
        for image,label in tqdm(zip(self.train_dataset.data,self.train_dataset.targets),total=len(self.train_dataset.data)):
            # Creazione Heat-Map per image
            image= transform_data_to_mask(image)
            image= torch.unsqueeze(image,0)
            image= image.to(self.device)
            
            hm = heat_map_generator(image)
           
        
            # Creazione Maschera
            mask_mean_value = np.nanmean(np.where(hm > 0, hm, np.nan), axis=(1, 2))[:, None, None]
            mask_std_value = np.nanstd(np.where(hm > 0, hm, np.nan), axis=(1, 2))[:, None, None]
            mask_threshold_value = mask_mean_value + 2 * mask_std_value
            mask = np.where(hm > mask_threshold_value, 0, 1)
                

            # Applicazione Maschera su immagine
            masked_image = np.array(image.cpu()) * mask
            masked_image = masked_image[0]
            masked_image = np.transpose(masked_image,(1,2,0))
            masked_image = 255 * masked_image # Now scale by 255
            masked_image = masked_image.astype(np.uint8)
            masked_data.append(masked_image)
            masked_labels.append(label)
            
        
        self.masked_dataset.data=np.asarray(masked_data)
        self.masked_dataset.targets=masked_labels

        self.masked_loader= torch.utils.data.DataLoader(
            self.masked_dataset,
            batch_size=128,
            sampler=self.train_sampler,
            shuffle=False,
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
                X, y = data[0], data[1]
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
    def train_erm(self,epochs=300, resume=False, best_resume_checkpoint_path: str=None, last_resume_checkpoint_path: str=None) -> None:
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
            #Update the scheduler
            self.lr_scheduler.step()
            self.logger.info(
                f"lr: {self.lr_scheduler.get_last_lr()[0]}",
                print_msg=True
            )

            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'accuracy' : val_accuracy,
                }, 'last_erm_model.pt')
            if(val_accuracy>self.best_accuracy):
                self.best_accuracy=val_accuracy
                torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'accuracy' : val_accuracy,
                }, 'best_erm_model.pt')


    #function used to test the accuracy of the model against the CelebA dataset
    def test(self, test_loader, checkpoint_path=None):
        self.logger.info("-" * 10 + "testing the model" +"-" * 10, print_msg=True)
        #LOAD THE MODEL SPECIFIED IN THE CHECKPOINT PATH
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        #RUN AN EPOCH IN TEST MODE AND USING A TEST LOADER SPECIFIED AS INPUT
        accuracy = self.run_an_epoch(
            data_loader=test_loader, epoch=epoch,device=self.device,mode="test"
        )
        self.logger.info("-" * 10 + f"Test accuracy ={accuracy}" +"-" * 10, print_msg=True)

        

#TO DO MaskData
#TO DO Test selective classification



        

            

