import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

image=plt.imread('../Imagebacteria.jpeg')
image = image[14:-14,30:-30,0] ## redimensionner + mettre sur une seule couche
print(image.shape)
plt.imshow(image,cmap='gray')
plt.show()

##en premier temps on va retire l'arriere plan on va donc utiliser du boolean indexing

#ici on regarde l'histogramme pour voir a partir de combien on va supprimer
image_2 = np.copy(image)
plt.hist(image_2.ravel(),bins=255)
plt.show()

image_filtre= image<160
plt.imshow(image_filtre,cmap='gray')
plt.show()

#on va maintenant enlever les quelques pixels qui reste (artefact)
open_x = ndimage.binary_opening(image_filtre)
plt.imshow(open_x,cmap='gray')
plt.show()

## on va maintenant utiliser label pour separer les bactéries et les compter

label_image, nb_labels = ndimage.label(open_x)
print(nb_labels)
plt.imshow(label_image)
plt.show()

#on va maintenant mesurer les bactéries
taille = ndimage.sum(open_x,label_image,range(nb_labels))
print(label_image[8])
plt.imshow(label_image==8)
plt.show()
print(taille)
plt.scatter(range(nb_labels),taille)
plt.show()
