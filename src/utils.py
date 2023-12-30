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


def save_model(folder: str, model):
    if not os.path.exists(folder):
        os.mkdir(folder)

    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"{model.__class__.__name__}_{time}.ckpt"

    full_path = os.path.join(folder, file_name)

    print(f"Model saved at \"{full_path}\"")

    torch.save(model.state_dict(), full_path)
