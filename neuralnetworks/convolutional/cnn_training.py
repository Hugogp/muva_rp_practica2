import os
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from neuralnetworks.convolutional.cnn import CNN
from src.CustomDataset import CustomDataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 5
batch_size = 100
learning_rate = 0.001

model = CNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_path = "../../images/train"

training_paths = []
training_labels = []

for training_dir in os.listdir(train_path):
    full_training_dir = os.path.join(train_path, training_dir)

    for training_image in os.listdir(full_training_dir):
        training_paths.append(os.path.join(full_training_dir, training_image))
        training_labels.append(training_dir)

train_data_paths, test_data_paths, train_labels, test_labels = train_test_split(
    training_paths, training_labels, test_size=0.2, random_state=42
)

train_dataset = CustomDataset(train_data_paths, train_labels)
test_dataset = CustomDataset(test_data_paths, test_labels)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# Define a mapping from class labels to numerical values
class_to_idx = {'forest': 0, 'fungus': 1, 'grass': 2, 'leaves': 3, 'salad': 4}

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)

        # Convert string labels to numerical indices
        labels = torch.tensor([class_to_idx[label] for label in labels]).to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        # images = images.reshape(-1, 150*150).to(device)
        images = images.view(-1, 3, 150, 150).to(device)

        # Convert string labels to numerical indices
        labels = torch.tensor([class_to_idx[label] for label in labels]).to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(len(test_data_paths), 100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model_cnn.ckpt')
