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
print(x.shape)
#model class

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        #Fully connected layers
        self.fc1 = nn.Linear(5*5*16, 120)
        #5*5*16 car on a 16 channels de 5*5
        #c'est le resuktat de la convolutiton ci dessus quand on print x.shape
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 10 car on a 10 classes
        #84 car on jour avec ce chiffre pour obtenir un bon resultat

    def forward(self, X):
        X= F.relu(self.conv1(X))
        X= F.max_pool2d(X, 2, 2)
        X= F.relu(self.conv2(X))
        X= F.max_pool2d(X, 2, 2)

        #flatten
        X= X.view(-1, 5*5*16)

        #Fully connected layers
        X= F.relu(self.fc1(X))
        X= F.relu(self.fc2(X))
        X= self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(41)
model = ConvolutionalNetwork()

#loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time
start_time = time.time()

#create a variable to keep track of the loss
epochs = 100
train_losses = []
test_losses = []
train_correct = []
test_correct = []

#For loop of epochs
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    #Train
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train) # prend les predictions du set d'entrainement et les met dans y_pred
        loss = criterion(y_pred, y_train) # compare les predictions avec les vrais valeurs

        predicted = torch.max(y_pred.data, 1)[1] # prend la valeur la plus probable
        batch_corr = (predicted == y_train).sum() # combien de lot a été correctement prédit
        trn_corr += batch_corr # ajoute le nombre de lot correctement prédit

        #Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print some results
        if b%600 == 0:
            print(f'epoch: {i} batch: {b} loss: {loss.item()} accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
    train_losses.append(loss)
    train_correct.append(trn_corr)

    #Test
    with torch.no_grad(): # on ne veut pas de gradient pour les tests car on ne veut pas entrainer le model
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

current_time = time.time()
total = current_time - start_time
print(f"Training took {total//60} minutes and {total%60} seconds")

#Graphs
train_losses= [tl.item() for tl in train_losses]
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend()
plt.show()

#Accuracy
plt.plot([t/600 for t in train_correct], label='training accuracy')
plt.plot([t/100 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()

test_load_everything = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_everything:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

print(correct.item()/10000)


torch.save(model.state_dict(), 'my_mnist_model.pt')

# Load the model
model2 = ConvolutionalNetwork()
model2.load_state_dict(torch.load('my_mnist_model.pt'))

