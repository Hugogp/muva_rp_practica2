import os
import cv2 as cv
import numpy as np
import torch
from natsort import natsorted
from torchvision.transforms import v2

from src.labels import index_to_label, get_labels_distribution
from src.neuralnetworks.AlexNet import AlexNet
from src.neuralnetworks.CNN import CNN
from src.neuralnetworks.cnn_extra_layers import CNNExtra


model_path = "./outputs/full/ReXNetV1_2024_01_02_20_03_35.pt"
# model_path = "./outputs/full/CNNExtra_2024_01_02_18_35_11.pt"
# model_path = "./outputs/full/AlexNet_2024_01_02_20_43_38.pt"
# path = "./outputs/CNNExtra_2023_12_31_12_48_00.ckpt"
test_dir = "./images/test"
images = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for image_path in natsorted(os.listdir(test_dir)):
    full_image_path = os.path.join(test_dir, image_path)

    if not image_path.endswith(".jpg"):
        continue

    image = cv.imread(full_image_path, cv.IMREAD_COLOR)

    image = cv.resize(image, (150, 150))

    if image.dtype == np.uint8():
        image = image.astype(np.float32()) / 255.0
    elif image.dtype != np.float32():
        image = image.astype(np.float32())

    image = v2.Compose([
        v2.ToImage(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(image)

    images.append(image)


model = torch.load(model_path)
model.eval()

images = torch.from_numpy(np.array(images)).view(-1, 3, 150, 150).to(device)

labels = []
for image in images:
    outputs = model(image.unsqueeze(0))

    _, predicted = torch.max(outputs.data, 1)
    prediction = index_to_label(predicted[0].item())

    labels.append(prediction)

print(get_labels_distribution(labels))

np.savetxt("Competicion2.txt", labels, fmt="%s", delimiter=",")
