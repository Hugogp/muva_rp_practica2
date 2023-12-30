import torch

from neuralnetworks.convolutional.cnn import CNN
from src.train_nn import train_nn

# Select device (use CUDA if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 5
batch_size = 100
learning_rate = 0.001

# Load Model
model = CNN(num_classes).to(device)

train_path = "images/train"
train_nn(model, train_path=train_path, epochs=num_epochs, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate)
