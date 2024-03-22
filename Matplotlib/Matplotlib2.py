# Top 5 scatter
import matplotlib.pyplot as plt

# scatter sert à faire des nuages de points pour pouvoir comparer deux variables

# exemple d'utilisation
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as load_iris

iris = load_iris.load_iris()
x = iris.data
y = iris.target
names = list(iris.target_names)

plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5, s=x[:, 2] * 100, label=names)
plt.xlabel('Longueur du sepal')
plt.ylabel('Largeur du sepal')
plt.show()

# probleme : on peut seulemen représenter deux variables sur les 4 disponibles


# Graphique en 3D
from mpl_toolkits.mplot3d import Axes3D

ax = plt.axes(projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
plt.show()

# graphique de surface en 3D
f = lambda x, y: np.sin(x) + np.cos(x + y)
X = np.linspace(0, 5, 100)
Y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# histogramme
plt.hist(x[:, 0], bins=20)

plt.show()

plt.hist2d(x[:, 0], x[:, 1], bins=20, cmap='Blues')
plt.xlabel('Longueur du sepal')
plt.ylabel('Largeur du sepal')
plt.show()

#contour plot

plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.show()

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
plt.show()

#image show

plt.imshow(np.corrcoef(x.T), cmap='Blues')
plt.colorbar()
plt.show()
#ca permet de voir la corrélation entre les variables


def graphiqueopti(data):
    n=len(data)
    plt.figure(figsize=(12,8))
    for k,i in zip(data.keys(),range(1,n+1)):
        plt.subplot(n,1,i)
        plt.plot(data[k])
        plt.title(k)
    plt.show()
