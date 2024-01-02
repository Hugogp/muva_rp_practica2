import os
import torch
from src.graphs import display_and_save_losses

from src.neuralnetworks.AlexNet import AlexNet
from src.neuralnetworks.CNN import CNN
from src.neuralnetworks.cnn_extra_layers import CNNExtra
from src.train_nn import train_nn
from src.utils import save_model, get_model_name, get_output_file_without_ext, save_hyperparameters


# Select device (use CUDA if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 2000
batch_size = 500
learning_rate = 1e-5

# Number of classes to classify
num_classes = 5

# Select the model
model = CNNExtra(num_classes).to(device)

# Train the model
print(f"Training \"{get_model_name(model)}\" on \"{device}\"...")

train_path = "images/train"
accuracy, total_test_images, losses = train_nn(model, train_path=train_path, epochs=num_epochs, batch_size=batch_size,
                                               learning_rate=learning_rate, device=device)

print('Accuracy of the network on the {} test images: {:.4f} %'.format(total_test_images, accuracy))

# Save the model
output_path = get_output_file_without_ext("./outputs", model)

print(f"Saving model & data at \"{output_path}\"")

if not os.path.exists("./outputs"):
    os.mkdir("./outputs")

# Save the model
save_model(output_path, model)

# Save hyperparameters in a .txt
save_hyperparameters(output_path, {
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "accuracy": accuracy,
})

# Display & save the loss graph
display_and_save_losses(losses, "{}: {:.4f}% accuracy".format(get_model_name(model), accuracy), f"{output_path}.png")
