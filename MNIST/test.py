from biased_mnist import *
from model import *
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torchvision

from tqdm import tqdm
from torch import Tensor


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, path: str, config: str) -> None:
        self.path = path
        if not config is None:
            with open(os.path.join(self.path, "log.txt"), "a") as f:
                f.write(config + "\n")

    def info(self, msg: str, print_msg: bool = False) -> None:
        if print_msg:
            print(msg)
        with open(os.path.join(self.path, "log.txt"), "a") as f:
            f.write(msg + "\n")

def calculate_accuracy(labels: torch.tensor, outputs: torch.tensor):
    equals = labels.eq(outputs)
    return torch.sum(equals).item() / len(labels)


class TrainBaseERM:
    def __init__(self):
      
      self.loss_function=nn.CrossEntropyLoss()
      self.model=SmallCNN(2)
      self.logger = Logger("", None)
      
      self.optimizer=optim.SGD(
                self.model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4
            )

    def run_an_epoch(self, data_loader, epoch, mode="train", device='cpu'):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        losses = AverageMeter()
        accuracies = AverageMeter()
        with torch.set_grad_enabled(mode == "train"):
            progress_bar = tqdm(data_loader)
            self.logger.info(
                f"{mode} epoch: {epoch}"
            )
            for data in progress_bar:
                progress_bar.set_description(f'{mode} epoch {epoch}')
                inputs, targets = data[0], data[2]
                inputs, targets = inputs.to(
                    device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                output_probabilities = F.softmax(outputs, dim=1)
                probabilities, predictions = output_probabilities.data.max(1)
                accuracies.update(calculate_accuracy(targets, predictions), 1)
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


if __name__ == '__main__':
    train_data_RGB = BiasedMNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True            
    )
    
    
    
    loaders = {
        'train' : torch.utils.data.DataLoader(train_data_RGB, 
                                              batch_size=128, 
                                              shuffle=True, 
                                              num_workers=1)
    }
    
    train= TrainBaseERM()
    
    for i in range(10):
        print("start one epoch")
        train.run_an_epoch(loaders['train'],i)
        print("-------------")