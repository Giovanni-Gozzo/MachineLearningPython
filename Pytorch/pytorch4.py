import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

transform = transforms.ToTensor()


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # Fully connected layers
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        # 5*5*16 car on a 16 channels de 5*5
        # c'est le resuktat de la convolutiton ci dessus quand on print x.shape
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 10 car on a 10 classes
        # 84 car on jour avec ce chiffre pour obtenir un bon resultat

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)

        # flatten
        X = X.view(-1, 5 * 5 * 16)

        # Fully connected layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


test_data = datasets.MNIST(root='cnn_data', train=False, download=True, transform=transform)
model = ConvolutionalNetwork()
model.load_state_dict(torch.load('my_mnist_model.pt'))

neufplusfin = test_data[4143][0]
neufplusfin = neufplusfin > 0.95
neufplusfin = neufplusfin.float()
plt.imshow(neufplusfin.reshape(28, 28), cmap='gray')
plt.show()

plt.imshow(test_data[4143][0].reshape(28, 28), cmap='gray')
plt.show()

model.eval()
with torch.no_grad():
    new_pred = model(test_data[4143][0].view(1, 1, 28, 28))
    new_pred2 = model(neufplusfin.view(1, 1, 28, 28))

print(new_pred.argmax())
print(new_pred2.argmax())

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Open the image
img4 = Image.open('IMG_2882.png')

# Apply the transformation
img4 = transform(img4)
img4 = 1 - img4

img4 = img4 > 0.5
img4 = img4.float()

plt.imshow(img4.squeeze().numpy(), cmap='gray')
plt.show()

model.eval()
with torch.no_grad():
    new_pred = model(img4.view(1, 1, 28, 28))

print(new_pred.argmax())
