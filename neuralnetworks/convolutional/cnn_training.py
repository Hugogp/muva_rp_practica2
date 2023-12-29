import os
from pathlib import Path
import torch
import torch.nn as nn
from neuralnetworks.convolutional.cnn import CNN
from src.CustomDataset import CustomDataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 5
batch_size = 100
learning_rate = 0.001

model = CNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_path = "../../images/train"
training_paths = [Path(os.path.join(train_path, folder)).as_posix() for folder in os.listdir(train_path)]

train_dataset = CustomDataset(training_paths)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)

        # Define a mapping from class labels to numerical values
        class_to_idx = {'forest': 0, 'fungus': 1, 'grass': 2, 'leaves': 3, 'salad': 4}

        # Convert string labels to numerical indices
        labels = torch.tensor([class_to_idx[label] for label in labels]).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'
           .format(epoch+1, num_epochs, loss.item()))
