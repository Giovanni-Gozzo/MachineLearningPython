#traitement d'image
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

#morphologie
np. random.seed(0)
X = np.zeros ((32, 32))
X[10:-10, 10:-10] = 1
X[np. random. randint(0,32,30),np.random.randint(0,32,30)] = 1
plt.imshow(X)
plt.show()

open_X = ndimage.binary_opening(X)
plt.imshow(open_X)
plt.show()

