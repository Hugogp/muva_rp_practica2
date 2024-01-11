import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import v2
from pathlib import Path


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, paths: [str], labels: [str]):
        self.images = paths
        self.labels = labels

        self.total_images = len(self.images)

    def __getitem__(self, index):
        image_path = Path(self.images[index]).as_posix()
        img = cv.imread(image_path, cv.IMREAD_COLOR)

        label = self.labels[index]

        img = CustomDataset.custom_normalize(img)

        img = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            # v2.RandomCrop((150, 150), padding=4),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(img)

        return img, label

    @staticmethod
    def custom_normalize(img) -> []:
        img = cv.resize(img, (150, 150))

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
