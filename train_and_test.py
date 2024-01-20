import os
import torch
from src.graphs import display_and_save_losses

from src.neuralnetworks.AlexNet import AlexNet
from src.neuralnetworks.CNN import CNN
from src.neuralnetworks.CoAtNet import coatnet_0, coatnet_1, coatnet_4
from src.neuralnetworks.RexNet import ReXNetV1
from src.neuralnetworks.cnn_extra_layers import CNNExtra
from src.train_nn import train_test_nn
from src.utils import save_model, get_model_name, get_output_file_without_ext, save_hyperparameters, save_training


# Select device (use CUDA if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 50
batch_size = 64
learning_rate = 1e-5


# Number of classes to classify
num_classes = 5

# Select the model
model = coatnet_0(num_classes).to(device)

# Train the model
print(f"Training \"{get_model_name(model)}\" on \"{device}\"...")

train_path = "images/train"
accuracy, total_test_images, losses, test_losses = train_test_nn(model, train_path=train_path, epochs=num_epochs,
                                                                 batch_size=batch_size, learning_rate=learning_rate,
                                                                 device=device)

print("Accuracy of the network on the {} test images: {:.4f}%".format(total_test_images, accuracy))

# Save the model
save_training(model, "./outputs", accuracy, losses, {
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "accuracy": accuracy,
}, test_losses)
