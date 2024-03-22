import numpy as np

##indexing

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(A[0, 0])
# ca donne le premier élément de la première ligne

print(A[0, :])
# ca donne la première ligne

print(A[:, 0])
# ca donne la première colonne

print(A[0:2, 0:2])
# ca donne les deux premières lignes et les deux premières colonnes

print(A[:, -2:])
# ca donne les deux dernières colonnes

B = np.zeros((4, 4))

B[1:-1, 1:-1] = 1
print(B)

C = np.zeros((5, 5))
C[::2, ::2] = 1
print(C)

D = np.random.randint(0, 10, (5, 5))
A[A < 5] = 10

print(A)

A[(A < 2) | (A == 10)] = 20
print(A)

from scipy import misc
import matplotlib.pyplot as plt

face = misc.face(gray=True)
plt.imshow(face,cmap=plt.cm.gray)
plt.show()

dimension=face.shape
dim1=(dimension[0]//8)
dim2=(dimension[1]//8)
face=face[dim1:-dim1,dim2:-dim2]
plt.imshow(face,cmap=plt.cm.gray)
plt.show()

face[face>200]=255
face[face<100]=0
plt.imshow(face,cmap=plt.cm.gray)
plt.show()


