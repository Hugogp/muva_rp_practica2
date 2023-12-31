import torch

from src.neuralnetworks.CNN import CNN
from src.train_nn import train_nn
from src.utils import save_model, get_model_name

# Select device (use CUDA if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
batch_size = 200
learning_rate = 0.00001

# Number of classes to classify
num_classes = 5

# Select the model
model = CNN(num_classes).to(device)

# Train the model
print(f"Training \"{get_model_name(model)}\" on \"{device}\"...")

train_path = "images/train"
accuracy, total_test_images = train_nn(model, train_path=train_path, epochs=num_epochs, batch_size=batch_size,
                                       learning_rate=learning_rate, device=device)

print('Accuracy of the network on the {} test images: {} %'.format(accuracy, total_test_images))

# Save the model
print("Saving model")

save_model("./outputs", model, {
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "accuracy": accuracy,
})
