import torch.nn as nn
import timm
#`timm` is a deep-learning library and a collection of SOTA computer vision models,
# layers, utilities, optimizers, schedulers, data-loaders, augmentations and also training/validating 
# scripts with ability to reproduce ImageNet training results.

#From Timm we import a pretrained resnet50. Pretrained on ImageNet
class ResNet50(nn.Module):
    def __init__(self, pretrained=False, num_classes=9):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
        
    def get_grad_cam_target_layer(self):
        return self.model.layer4[-1]

    def forward(self, x):
        return self.model(x)