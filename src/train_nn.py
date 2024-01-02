import time
import torch
import torch.nn as nn

from src.utils import get_dataloaders


def train_nn(model, train_path: str, epochs: int, batch_size: int, learning_rate: float, device) -> (float, int, [float]):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = get_dataloaders(train_path, batch_size)

    losses = _run_training(train_loader, model=model, num_epochs=epochs, device=device, criterion=criterion, optimizer=optimizer)

    accuracy, total_images_test = _calculate_test_score(test_loader, model=model, device=device)

    return accuracy, total_images_test, losses


def _run_training(train_loader, num_epochs, device, model, criterion, optimizer) -> [float]:
    loss = None
    losses = []

    for epoch in range(num_epochs):
        start_time = time.time()

        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_taken = time.time() - start_time

        losses.append(loss.item())

        print('Epoch [{}/{}], Loss: {:.4f} in {:.2f} seconds'.format(epoch + 1, num_epochs, loss.item(), time_taken))

    return losses


def _calculate_test_score(test_loader, device, model) -> (float, int):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            # images = images.reshape(-1, 150*150).to(device)
            images = images.view(-1, 3, 150, 150).to(device)

            # Convert string labels to numerical indices
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, total
