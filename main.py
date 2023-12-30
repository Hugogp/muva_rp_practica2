import os
from pathlib import Path
from neuralnetworks.resnet import ResNet
from neuralnetworks.resnet import ResidualBlock
from neuralnetworks.convolutional.cnn import CNN
from neuralnetworks.resnet_custom import ResNet18
from src.CustomDataset import CustomDataset
from src.train_nn import train_nn
import os
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 5
batch_size = 10
learning_rate = 0.001

model = ResNet18(num_classes)
train_path = "images/train"

train_nn(model, train_path=train_path, epochs=num_epochs, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate)
