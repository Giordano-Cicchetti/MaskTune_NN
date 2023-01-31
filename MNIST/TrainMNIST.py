import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import *
from others import *
from biased_mnist import *
import math 
from numpy.random import default_rng
from copy import deepcopy
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from pytorch_grad_cam import XGradCAM
from torchvision.utils import save_image

class TrainMNIST:
    def __init__(self,device):
        #initialize all is needed
        #DEVICE
        self.device       = device
        #LOSS FUNCTION
        self.loss_function= nn.CrossEntropyLoss()
        #MODEL
        self.model        = CNN_MNIST(num_classes=2)
        self.model        = self.model.to(self.device)
        #LOGGER
        self.logger       = Logger("", None)
        #OPTIMIZER
        self.optimizer    = optim.SGD(
                self.model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4
            )
        #SCHEDULER 
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=[25,50,75],
                gamma= 0.5,
                last_epoch=-1,
            )
        #And now we can initialize all the DATASETS and DATALOADERS:
        self.initialize_datasets_loaders()
    
    def th_delete(self,tensor, indices):
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]

    def initialize_datasets_loaders(self):
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
                target = int(self.train_dataset.targets[val_index])
                new_file_path = os.path.join(
                    val_data_dir, f"{val_index}{target}.png")
                os.replace(file_path, new_file_path)

            self.train_dataset.data_new=[]
            self.train_dataset.data_new_paths=[]
            self.train_dataset.targets=self.th_delete(self.train_dataset.targets,val_indices)


            image_file_paths = glob(
                os.path.join(self.train_dataset.img_data_dir, "*")
            )
            self.train_dataset.data_new_paths += image_file_paths
            for image_path in image_file_paths:
                temp = Image.open(image_path)
                keep = temp.copy()
                self.train_dataset.data_new.append(keep)
                temp.close()

        self.val_dataset.targets=[]
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
            target=int(image_path.split(".")[-2][-1])
            self.val_dataset.targets.append(target)      
        self.val_dataset.targets=torch.Tensor(self.val_dataset.targets)
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


    #####
    
    def mask_data(self, train_loader,erm_checkpoint_path: str=None):
        #Load the trained model for masking the image
        if(erm_checkpoint_path!=None):
          checkpoint = torch.load(erm_checkpoint_path)
          self.model.load_state_dict(checkpoint['model_state_dict'])
          self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          
        
        heat_map_generator = XGradCAM(
            model=self.model,
            target_layers=[self.model.get_grad_cam_target_layer()],
            use_cuda=True,
        )
      
        self.masked_dataset = deepcopy(self.train_dataset)
        masked_data_dir = self.train_dataset.img_data_dir.replace("train", "masked")
        if not (os.path.isdir(masked_data_dir) and len(os.listdir(masked_data_dir)) > 0):
            os.makedirs(masked_data_dir, exist_ok=True)

            counter_imgs = 0
            #maskedimg = torch.empty(1,3,28,28)
            for data in tqdm(train_loader):
                # Creazione Heat-Map per batch
                i1,i2,i3 = data[0],data[1],data[2]
                hm = heat_map_generator(i1)
        
                # Creazione Maschera
                mask_mean_value = np.nanmean(np.where(hm > 0, hm, np.nan), axis=(1, 2))[:, None, None]
                mask_std_value = np.nanstd(np.where(hm > 0, hm, np.nan), axis=(1, 2))[:, None, None]
                mask_threshold_value = mask_mean_value + 2 * mask_std_value
                masks = np.where(hm > mask_threshold_value, 0, 1)

                # Applicazione Maschera su immagini del batch
                
                for image,mask,target in zip(data[0],masks,data[2]):
          
                    masked_images = image*mask
                    masked_images.numpy()
                    target=int(target)
                    save_image(masked_images, os.path.join(masked_data_dir, f"{counter_imgs}{target}.png"))

                    counter_imgs += 1
          
        
        #create variables data_new e data_new_paths
        data_new=[]
        data_new_paths=[]      
        targets = []

        image_file_paths = glob(
            os.path.join(masked_data_dir, "*")
        )
        data_new_paths += image_file_paths
        for image_path in image_file_paths:
            temp = Image.open(image_path)
            keep = temp.copy()
            data_new.append(keep)
            temp.close()
            target=int(image_path.split(".")[-2][-1])
            targets.append(target) 
        self.masked_dataset.targets=torch.Tensor(targets)
        self.masked_dataset.data_new=data_new
        self.masked_dataset.data_new_paths=data_new_paths

        #Creation of masked data loader
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

    #Function used to train the entire convolutional network for a certan number of epochs
    def train_erm(self, epochs=100, best_resume_checkpoint_path: str=None, last_resume_checkpoint_path: str=None) -> None:
        resume_epoch = 0
        self.best_accuracy=-math.inf
        #If a checkpoint is specified then restart the train
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

    #Function used to test the model by using a test loader specified as parameter
    def test(self, test_loader,checkpoint_path=None):
        self.logger.info("-" * 10 + "testing the model" +"-" * 10, print_msg=True)
        #LOAD THE MODEL SPECIFIED IN THE CHECKPOINT PATH
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        #RUN AN EPOCH IN TEST MODE AND USING A TEST LOADER SPECIFIED AS INPUT
        accuracy = self.run_an_epoch(
            data_loader=test_loader, epoch=epoch, mode="test",device=self.device
        )
        self.logger.info("-" * 10 + f"Test accuracy ={accuracy}" +"-" * 10, print_msg=True)

        

            

