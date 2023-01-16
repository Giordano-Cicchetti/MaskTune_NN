import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import *
from others import *

class TrainBaseERM:
    def __init__(self,device):
      self.device       = device
      self.loss_function= nn.CrossEntropyLoss()
      self.model        = SmallCNN(2)
      self.model        = self.model.to(self.device)
      self.logger       = Logger("", None)
      self.optimizer    = optim.SGD(
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
