import os
import torch
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.CustomDataset import CustomDataset


def get_dataloaders(train_path: str, batch_size: int):
    training_paths = []
    training_labels = []

    # Define a mapping from class labels to numerical values
    class_to_idx = {'forest': 0, 'fungus': 1, 'grass': 2, 'leaves': 3, 'salad': 4}

    for training_dir in os.listdir(train_path):
        full_training_dir = os.path.join(train_path, training_dir)

        for training_image in os.listdir(full_training_dir):
            if not training_image.endswith(".jpg"):
                continue

            training_paths.append(os.path.join(full_training_dir, training_image))
            training_labels.append(class_to_idx[training_dir])

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

    return train_loader, test_loader


def save_model(file_path: str, model):
    torch.save(model.state_dict(), f"{file_path}.ckpt")


def save_hyperparameters(file_path: str, hyperparameters: dict):
    with open(f"{file_path}.txt", "w") as file:
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")


def get_output_file_without_ext(folder: str, model) -> str:
    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"{get_model_name(model)}_{time}"

    return os.path.join(folder, file_name)


def get_model_name(model) -> str:
    return model.__class__.__name__
