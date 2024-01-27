import os
import cv2 as cv
import numpy as np
import torch
from natsort import natsorted
from torchvision.transforms import v2

from src.CustomDataset import CustomDataset
from src.constants import IMAGES_HEIGHT, IMAGES_WIDTH
from src.labels import index_to_label, get_labels_distribution
from src.neuralnetworks.AlexNet import AlexNet
from src.neuralnetworks.CNN import CNN
from src.neuralnetworks.cnn_extra_layers import CNNExtra


model_path = "outputs/full/CoAtNet_2024_01_16_16_37_47.pt"
test_dir = "./images/test"
images = []

print("Starting up testing...")

# Select the device to use
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Prepare all the images (in the correct order)
for image_path in natsorted(os.listdir(test_dir)):
    full_image_path = os.path.join(test_dir, image_path)

    # Skip files that are not images
    if not image_path.endswith(".jpg"):
        continue

    image = cv.imread(full_image_path, cv.IMREAD_COLOR)
    image = CustomDataset.custom_normalize(image)

    image = v2.Compose([
        v2.ToImage(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(image)

    images.append(image)

# Load the model
model = torch.load(model_path)
model.eval()

# Move the images to the device
images = torch.from_numpy(np.array(images)).view(-1, 3, IMAGES_WIDTH, IMAGES_HEIGHT).to(device)

# Calculate each model prediction
labels = []
for image in images:
    outputs = model(image.unsqueeze(0))

    _, predicted = torch.max(outputs.data, 1)
    prediction = index_to_label(predicted[0].item())

    labels.append(prediction)

# Show the distribution of the labels
# We have prior knowledge that the distribution should be around 100 for each class
print(get_labels_distribution(labels))

# Save the file
np.savetxt("Competicion2.txt", labels, fmt="%s", delimiter=",")
