import torch
import torch.nn as nn

from src.utils import get_dataloaders


def train_nn(model, train_path: str, epochs: int, num_classes: int, batch_size: int, learning_rate: float):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = get_dataloaders(train_path, batch_size)

    _run_training(train_loader, model=model, num_epochs=epochs, device=device, criterion=criterion, optimizer=optimizer)

    _display_test_score(test_loader, model=model, device=device)


def _run_training(train_loader, num_epochs, device, model, criterion, optimizer):
    loss = None

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            # Convert string labels to numerical indices
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


def _display_test_score(test_loader, device, model):
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

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))
