import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import v2
from pathlib import Path

from src.constants import IMAGES_WIDTH, IMAGES_HEIGHT


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset to handle the structure of the files and folders for this project.
    """
    def __init__(self, paths: [str], labels: [str]):
        self.images = paths
        self.labels = labels

        self.total_images = len(self.images)

    def __getitem__(self, index):
        image_path = Path(self.images[index]).as_posix()
        img = cv.imread(image_path, cv.IMREAD_COLOR)

        label = self.labels[index]

        # Normalize the images using OpenCV
        img = CustomDataset.custom_normalize(img)

        # Transform the images using PyTorch
        img = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])(img)

        return img, label

    @staticmethod
    def custom_normalize(img) -> []:
        """
        Makes sure the image provided is 150x150 (IMAGES_WIDTHxIMAGES_HEIGHT) and then convert it to float32
        """
        img = cv.resize(img, (IMAGES_WIDTH, IMAGES_HEIGHT))

        if img.dtype == np.uint8():
            img = img.astype(np.float32()) / 255.0
        elif img.dtype != np.float32():
            img = img.astype(np.float32())

        return img

    def __len__(self):
        return self.total_images

    @staticmethod
    def get_label(image_path):
        return image_path.split("/")[-1]
