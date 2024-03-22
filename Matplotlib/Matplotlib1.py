import numpy as np

x = np.linspace(0, 2, 10)
y = x ** 2
print(x)

# pyplot
import matplotlib.pyplot as plt

plt.plot(x, y, c='red')
plt.show()

plt.scatter(x, y)
plt.show()

# plt.plot(x,y,c='red',ls='--',lw=2,label='line')
# c: color
# ls: line style
# lw: line width
# label: legend


# Cycle de vie d'une figure
fig = plt.figure()  # debut de la figure
plt.plot(x, y, c='red', ls='--', lw=2, label='quadratique')
plt.plot(x, x ** 3, label='cubique')
plt.title('Mon premier graphique')
plt.xlabel('axe des x')
plt.ylabel('axe des y')
plt.legend()
plt.show()
plt.savefig('figure1.png')  # sauvegarder la figure

# Subplots
plt.subplot(2, 1, 1)
plt.plot(x, y, c='red')
plt.subplot(2, 1, 2)
plt.plot(x, x ** 3)

# exercice

dataset = {f"experience{i}": np.random.randn(100) for i in range(4)}


def graphique(dataset):
    fig = plt.figure()
    axesx = np.arange(100)
    for i in range(len(dataset)):
        plt.subplot(len(dataset), 1, i + 1)
        plt.plot(axesx, dataset["experience" + str(i)])
    plt.show()


graphique(dataset)

def graphiqueopti(data):
    n=len(data)
    plt.figure(figsize=(12,8))
    for k,i in zip(data.keys(),range(1,n+1)):
        plt.subplot(n,1,i)
        plt.plot(data[k])
        plt.title(k)
    plt.show()
graphiqueopti(dataset)
