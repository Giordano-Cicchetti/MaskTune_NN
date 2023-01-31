import torch.nn as nn


class CNN_MNIST(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        #First convolution block: Conv2d + batchNorm + Relu
        self.conv1 = nn.Sequential(
            #input: images 28x28x3 
            #output: images 26x26x16
            nn.Conv2d(3, 16, (3, 3), (1, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
            )
        
        #Second convolution block: Conv2d + batchNorm + Relu + MaxPool
        self.conv2 = nn.Sequential(
            #Input: images 26x26x16
            #Output: images 12x12x16
            nn.Conv2d(16, 16, (3, 3), (1, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2))
            )
        #Third convolution block: Conv2d + batchNorm + Relu
        self.conv3 = nn.Sequential(
            #Input: images 12x12x16
            #Output: images 10x10x32 
            nn.Conv2d(16, 32, (3, 3), (1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
            )
        #Fourth convolution block: Conv2d + batchNorm + Relu + MaxPool
        self.conv4 = nn.Sequential(
            #Input: images 10x10x32
            #Output: images 4x4x32
            nn.Conv2d(32, 32, (3, 3), (1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            )
        #Take images of shape 4x4x32 and flattened them into a tensor of 512 values
        self.flatten= nn.Flatten()

        #Sequence of two fully connected layers 
        self.linear = nn.Sequential(
            #Input: Flattened images of 512 values
            #Output: 2 logits
            nn.Linear(512, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )

    def get_grad_cam_target_layer(self):
        return self.conv4[-2]

    def forward(self, x):
        #Pass inputs to the four conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        
        #Pass the extracted features to the fully connected classifier
        logits = self.linear(x)

        return logits