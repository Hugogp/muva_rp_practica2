import os
import cv2 as cv
import numpy as np
import torch
from natsort import natsorted

from src.neuralnetworks.AlexNet import AlexNet
from src.neuralnetworks.CNN import CNN
from src.neuralnetworks.cnn_extra_layers import CNNExtra

# PATH = "./outputs/CNNExtra_2023_12_31_12_48_00.ckpt"
PATH = "./outputs/AlexNet_2024_01_02_13_59_38.ckpt"
ipath = "./images/test"
images = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

idx = {0: 'forest', 1: 'fungus', 2: 'grass', 3: 'leaves', 4: 'salad'}
dicc = {'forest': 0, 'fungus': 0, 'grass': 0, 'leaves': 0, 'salad': 0}
for image_path in natsorted(os.listdir(ipath)):
    full_image_path = os.path.join(ipath, image_path)

    if not image_path.endswith(".jpg"):
        continue

    image = cv.imread(full_image_path, cv.IMREAD_COLOR)

    image = cv.resize(image, (150, 150))

    if image.dtype == np.uint8():
        image = image.astype(np.float32()) / 255.0
    elif image.dtype != np.float32():
        image = image.astype(np.float32())

    print(np.max(image), np.min(image))

    images.append(image)

model = AlexNet(num_classes=5).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

images = torch.from_numpy(np.array(images)).view(-1, 3, 150, 150).to(device)

labels = []
for image in images:
    outputs = model(image)
    print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    prediction = idx[predicted[0].item()]

    labels.append(prediction)

np.savetxt("Competicion2.txt", labels, fmt="%s", delimiter=",")
for label in labels:
    dicc[label] += 1
print(dicc)
