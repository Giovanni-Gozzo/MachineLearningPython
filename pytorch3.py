# CONVOLUTIONAL NEURAL NETWORK
import matplotlib
# https://setosa.io/ev/image-kernels/

import matplotlib.pyplot as plt
import numpy as np

image = plt.imread('./image0000001 3.JPG')

image = image[:, :, 0]
plt.imshow(image, cmap='gray')
plt.show()


def convolution(img, kernel):
    image = np.copy(img)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            image[i, j] = image[i - 1, j - 1] * kernel[0] + image[i - 1, j] * kernel[1] + image[i - 1, j + 1] * kernel[
                2] + \
                          image[i, j - 1] * kernel[3] + image[i, j] * kernel[4] + image[i, j + 1] * kernel[5] + image[
                              i + 1, j - 1] * kernel[6] + image[i + 1, j] * kernel[7] + image[i + 1, j + 1] * kernel[8]
    return image


kernel = np.repeat(1 / 9, 9)
images_new = convolution(image, kernel)
plt.imshow(images_new, cmap='gray')
plt.show()

# Ca marche pas bien (test de giogio)

# commencons l'autre vidéo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#convert image files to tensor of 4 dimensions (nb of images, color, width, height)
transform = transforms.ToTensor()

#Train Data
train_date = datasets.MNIST(root='cnn_data', train=True, download=True, transform=transform)
print(train_date)

#Test Data
test_data = datasets.MNIST(root='cnn_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_date, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# Define our CNN model

conv1= nn.Conv2d(1, 6, 3, 1)
conv2= nn.Conv2d(6, 16, 3, 1)
for i,(X_train, y_train) in enumerate(train_date):
    break

x = X_train.view(1, 1, 28, 28)

x= F.relu(conv1(x))
# la on applique notre premiere couche de convolution
x= F.max_pool2d(x, 2, 2)
# la ca va divise l'image par 2 car on a un kernel de 2 et un stride de 2 (on prend un pixel sur 2) et on a un padding de 0

x= F.relu(conv2(x))
# Sans definir de padding, le padding est de 0 et on perd donc 2 pixels sur chaque coté
x= F.max_pool2d(x, 2, 2)

