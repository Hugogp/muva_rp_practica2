import time
import torch
import torch.nn as nn

from src.constants import IMAGES_WIDTH, IMAGES_HEIGHT
from src.utils import get_dataloaders, get_training_data, generate_data_loader


def train_test_nn(model, train_path: str, epochs: int, batch_size: int, learning_rate: float, device) -> (float, int, [float], [float]):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = get_dataloaders(train_path, batch_size)

    # Run training and obtain losses
    losses, test_losses = _run_training(train_loader, model=model, num_epochs=epochs, device=device, criterion=criterion,
                                        optimizer=optimizer, test_loader=test_loader)

    # Calculate test accuracy and total images tested
    accuracy, total_images_test = _calculate_test_score(test_loader, model=model, device=device)

    return accuracy, total_images_test, losses, test_losses


def only_train_nn(model, train_path: str, epochs: int, batch_size: int, learning_rate: float, device) -> [float]:
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_paths, training_labels = get_training_data(train_path)

    # Generate the data loader for training
    train_loader = generate_data_loader(training_paths, training_labels, batch_size)

    # Get the loads
    losses, _ = _run_training(train_loader, model=model, num_epochs=epochs, device=device, criterion=criterion,
                              optimizer=optimizer, test_loader=None)

    return losses


def _run_training(train_loader, num_epochs, device, model, criterion, optimizer, test_loader=None) -> [float]:
    """
    Runs the training for the given model and with the given parameters.
    Will also try to generate a test score if the test_loader is provided.
    """
    loss = None
    losses = []
    test_losses = [] if test_loader else None

    model.train()

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

        print("Epoch [{}/{}], Loss: {:.4f} in {:.2f} seconds".format(epoch + 1, num_epochs, loss.item(), time_taken))

        # If the test_loader is provided, use it to calculate the Test Loss
        if test_loader:
            test_start_time = time.time()
            test_loss = _calculate_loss_score(model, test_loader, criterion, device)
            test_time = time.time() - test_start_time

            test_losses.append(test_loss)

            print("[TEST] Epoch [{}/{}], Loss: {:.4f} in {:.2f} seconds".format(epoch + 1, num_epochs, test_loss, test_time))

    return losses, test_losses


def _calculate_test_score(test_loader, device, model) -> (float, int):
    """
    Calculate the test score for the model.
    """
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.view(-1, 3, IMAGES_WIDTH, IMAGES_HEIGHT).to(device)

            # Convert string labels to numerical indices
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, total


def _calculate_loss_score(model, test_loader, criterion, device) -> float:
    """
    Calculate the loss score for the model.
    """
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for test_image, test_label in test_loader:
            predictions = model(test_image.to(device))

            loss = criterion(predictions, test_label.to(device))

            total_loss += loss.item()

    model.train()

    return total_loss / len(test_loader)
