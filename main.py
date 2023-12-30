
import torch

from neuralnetworks.convolutional.cnn import CNN
from neuralnetworks.resnet_custom import ResNet18
from src.train_nn import train_nn
from src.utils import save_model

# Select device (use CUDA if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 5
batch_size = 100
learning_rate = 0.001

model = ResNet18(num_classes).to(device)

train_path = "images/train"
train_nn(model, train_path=train_path, epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, device=device)

save_model("./outputs", model)
