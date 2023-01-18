import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import *
from others import *
from biased_mnist import *
import math 
import shutil
from numpy.random import default_rng
from copy import deepcopy
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

class TrainBaseERM:
    def __init__(self,device):
        #initialize all what is needed
        self.device       = device
        self.loss_function= nn.CrossEntropyLoss()
        self.model        = SmallCNN(2)
        self.model        = self.model.to(self.device)
        self.logger       = Logger("", None)
        #Optimizer
        self.optimizer    = optim.SGD(
                self.model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4
            )
        #Scheduler 
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=[25,50,75],
                gamma= 0.5,
                last_epoch=-1,
            )
        #And now we can initialize all the datasets and dataloaders:
        #TRAIN DATASET
        self.train_dataset = BiasedMNIST(
                root = 'data',
                train = True,                         
                transform = ToTensor(), 
                download = True,            
            )
        #VALIDATION DATASET
        self.val_dataset = deepcopy(self.train_dataset)
        val_data_dir = self.train_dataset.img_data_dir.replace("train", "val")
        if not (os.path.isdir(val_data_dir) and len(os.listdir(val_data_dir)) > 0):
            
            os.makedirs(val_data_dir, exist_ok=True)
            rng = default_rng()
            val_indices = rng.choice(len(self.train_dataset), size=12000, replace=False)
            for val_index in val_indices:
                file_path = self.train_dataset.data_new_paths[val_index]
                target = self.train_dataset.targets[val_index]
                new_file_path = os.path.join(
                    val_data_dir, f"{val_index}.png")
                os.replace(file_path, new_file_path)
            self.train_dataset.data_new=[]
            self.train_dataset.data_new_paths=[]
        
            image_file_paths = glob(
                os.path.join(self.train_dataset.img_data_dir, "*")
            )
            self.train_dataset.data_new_paths += image_file_paths
            for image_path in image_file_paths:
                temp = Image.open(image_path)
                keep = temp.copy()
                self.train_dataset.data_new.append(keep)
                temp.close()

        self.val_dataset.data_new=[]
        self.val_dataset.data_new_paths=[]
    
        image_file_paths = glob(
            os.path.join(val_data_dir, "*")
        )
        self.val_dataset.data_new_paths += image_file_paths
        for image_path in image_file_paths:
            temp = Image.open(image_path)
            keep = temp.copy()
            self.val_dataset.data_new.append(keep)
            temp.close()           


        #TEST DATASET WITH ORIGINAL DATA
        self.test_dataset_original = BiasedMNIST(
                root = 'data',
                train = False,
                biased= False,                         
                transform = ToTensor(), 
                download = True,            
            )
        #TEST DATASET WITH BIASED DATA
        self.test_dataset_biased = BiasedMNIST(
                root = 'data',
                train = False,
                biased= True,                         
                transform = ToTensor(), 
                download = True,            
            )
        #TRAIN LOADER
        self.train_loader =  torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=1
        )
        #VALIDATION LOADER
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=1
        )
        #TEST LOADERS: ORIGINAL AND BIASED
        self.test_loader_original = torch.utils.data.DataLoader(
            self.test_dataset_original,
            batch_size=128,
            shuffle=False,
            num_workers=1
        )
        self.test_loader_biased = torch.utils.data.DataLoader(
            self.test_dataset_biased,
            batch_size=128,
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

    #Function used to train the entire convolutional network for a certan number of epochs
    def train_erm(self,epochs=100, resume=False, best_resume_checkpoint_path: str=None, last_resume_checkpoint_path: str=None) -> None:
        resume_epoch = 0
        self.best_accuracy=-math.inf
        if(best_resume_checkpoint_path!=None and last_resume_checkpoint_path!=None):
            checkpoint = torch.load("last_erm_model.pt")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            resume_epoch = checkpoint['epoch'] + 1

            checkpoint = torch.load("best_erm_model.pt")
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


    def test(self, test_loader,checkpoint_path=None):
        self.logger.info("-" * 10 + "testing the model" +
                         "-" * 10, print_msg=True)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy = self.run_an_epoch(
            data_loader=test_loader, epoch=epoch, mode="test",device=self.device
        )

        print(f"Test accuracy = {accuracy}")
        

            

