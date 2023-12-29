import os
from pathlib import Path
from src.CustomDataset import CustomDataset

train_path = "images/train"
training_paths = [Path(os.path.join(train_path, folder)).as_posix() for folder in os.listdir(train_path)]

train_dataset = CustomDataset(training_paths)
test_dataset = CustomDataset(["images/test"])
