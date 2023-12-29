import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2 as cv


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, paths: [str]):
        self.images = []
        self.labels = []

        for path in paths:
            contents = os.listdir(path)
            self.images.extend([os.path.join(path, content_path) for content_path in contents])
            self.labels.extend([self.get_label(path)] * len(contents))

        self.total_images = len(self.images)

    def __getitem__(self, index):
        image_path = Path(self.images[index]).as_posix()
        img = cv.imread(image_path, cv.IMREAD_COLOR)

        label = self.labels[index]

        img = cv.resize(img, (150, 150))

        if img.dtype == np.uint8():
            img = img.astype(np.float32()) / 255.0
        elif img.dtype != np.float32():
            img = img.astype(np.float32())

        img = transforms.ToTensor()(img)

        return img, label

    def __len__(self):
        return self.total_images

    @staticmethod
    def get_label(image_path):
        return image_path.split("/")[-1]
