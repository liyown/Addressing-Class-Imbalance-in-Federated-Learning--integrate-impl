# McMahan et al., 2016; 1,663,370 parameters
import copy
from collections import OrderedDict
import torch.nn.functional as F
import torch
from torch import nn


class FedAvgCNN(nn.Module):
    def __init__(self, dataset: str = "mnist"):
        super(FedAvgCNN, self).__init__()
        config = {
            "mnist": (1, 1024, 10),
            "fmnist": (1, 1024, 10),
            "cifar10": (3, 1600, 10),
            "cifar100": (3, 1600, 100),
        }
        self.features = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                flatten=nn.Flatten(),
                linear1=nn.Linear(config[dataset][1], 512),
                activation1=nn.ReLU(),
                linear2=nn.Linear(512, config[dataset][2]),
            )
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.features(x))



if __name__ == '__main__':

    net = FedAvgCNN()
    print()