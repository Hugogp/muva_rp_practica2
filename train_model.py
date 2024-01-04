import torch

from src.neuralnetworks.AlexNet import AlexNet
from src.neuralnetworks.CNN import CNN
from src.neuralnetworks.RexNet import ReXNetV1
from src.neuralnetworks.cnn_extra_layers import CNNExtra
from src.train_nn import only_train_nn
from src.utils import get_model_name, save_training


# Select device (use CUDA if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 1000
batch_size = 512
learning_rate = 1e-5

# Number of classes to classify
num_classes = 5

# Select the model
model = AlexNet(num_classes).to(device)

# Train the model
print(f"Training \"{get_model_name(model)}\" on \"{device}\"...")

train_path = "images/train"
losses = only_train_nn(model, train_path=train_path, epochs=num_epochs, batch_size=batch_size,
                       learning_rate=learning_rate, device=device)

# Save the model
save_training(model, "./outputs/full", -1, losses, {
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "accuracy": -1,
})
