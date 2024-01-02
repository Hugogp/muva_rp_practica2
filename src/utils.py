import os
import torch
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.CustomDataset import CustomDataset
from src.graphs import display_and_save_losses
from src.labels import label_to_index


def get_dataloaders(train_path: str, batch_size: int):
    training_paths, training_labels = get_training_data(train_path)

    train_data_paths, test_data_paths, train_labels, test_labels = train_test_split(
        training_paths, training_labels, test_size=0.2, random_state=42
    )

    train_loader = generate_data_loader(train_data_paths, train_labels, batch_size)
    test_loader = generate_data_loader(test_data_paths, test_labels, batch_size)

    return train_loader, test_loader


def generate_data_loader(data_paths: [str], labels: [int], batch_size: int):
    dataset = CustomDataset(data_paths, labels)

    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True)


def get_training_data(train_path: str) -> ([str], [int]):
    training_paths = []
    training_labels = []

    for training_dir in os.listdir(train_path):
        full_training_dir = os.path.join(train_path, training_dir)

        for training_image in os.listdir(full_training_dir):
            if not training_image.endswith(".jpg"):
                continue

            training_paths.append(os.path.join(full_training_dir, training_image))
            training_labels.append(label_to_index(training_dir))

    return training_paths, training_labels


def save_training(model, folder: str, accuracy: float, losses: [float], hyperparameters: dict):
    output_path = get_output_file_without_ext(folder, model)

    print(f"Saving model & data at \"{output_path}\"")

    if not os.path.exists(folder):
        os.mkdir(folder)

    # Save the model
    save_model(output_path, model)

    # Save hyperparameters in a .txt
    save_hyperparameters(output_path, hyperparameters)

    # Display & save the loss graph
    display_and_save_losses(losses, "{}: {:.4f}% accuracy".format(get_model_name(model), accuracy),
                            f"{output_path}.png")


def save_model(file_path: str, model):
    torch.save(model, f"{file_path}.pt")


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
