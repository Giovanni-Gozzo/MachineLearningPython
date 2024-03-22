import matplotlib.pyplot as plt
import numpy as np

# Module interpolate
x = np.linspace(0, 10, 10)
y = x ** 2

from scipy.interpolate import interp1d

f = interp1d(x, y, kind='linear')

# il y a plusieurs types de fonctions d'interpolation
# 'linear' : interpolation linéaire
# 'nearest' : interpolation par la valeur la plus proche
# 'zero' : interpolation par la valeur nulle
# 'slinear' : interpolation linéaire
# 'quadratic' : interpolation parabolique
# 'cubic' : interpolation cubique
# 'previous' : interpolation par la valeur précédente
# 'next' : interpolation par la valeur suivante

new_x = np.linspace(0, 10, 30)
result = f(new_x)

plt.scatter(x, y)
plt.scatter(new_x, result, color='red')
plt.show()

# Module optimize
from scipy import optimize, fftpack

x = np.linspace(0, 2, 100)
y = 1 / 3 * x ** 3 - 3 / 5 * x ** 2 + 2 + np.random.randn(x.shape[0]) / 20
plt.scatter(x, y)
plt.show()


def f(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


params, param_cov = optimize.curve_fit(f, x, y)

plt.scatter(x, y)
plt.plot(x, f(x, params[0], params[1], params[2], params[3]), color='red')
plt.show()


# minimisation

def f(x):
    return x ** 2 + 15 * np.sin(x)


x = np.linspace(-10, 10, 100)
plt.plot(x, f(x))
plt.show()

result = optimize.minimize(f, x0=-5).x

plt.plot(x, f(x))
plt.scatter(result, f(result), color='red')
plt.show()

# traitement du signal
x = np.linspace(0, 20, 100)
y = x + 4 * np.sin(x) + np.random.randn(x.shape[0])
plt.plot(x, y)

from scipy import signal

new_y = signal.detrend(y)
plt.plot(x, new_y)
plt.show()

# transformée de Fourier
x = np.linspace(0, 30, 1000)
y = 3 * np.sin(x) + 2 * np.sin(5 * x) + np.sin(10 * x)

from scipy import fftpack

fourier = fftpack.fft(y)
power = np.abs(fourier)
frequence = fftpack.fftfreq(len(y))
plt.plot(np.abs(frequence), power)
plt.show()

# la transformaton de fourrier sert a filtre des signal par exemple
# pour enlever le bruit

# en gros on prend le signal de base on applique la transformée de fourrier
# apres on le filtre avec du boolean indexing pour enlever les bruits trop fort
# et on applique la transformée de fourrier inverse pour retrouver le signal de base

# exemple
x = np.linspace(0, 30, 1000)
y = 3 * np.sin(x) + 2 * np.sin(5 * x) + np.sin(10 * x) + np.random.randn(x.shape[0])
fourier = fftpack.fft(y)
power = np.abs(fourier)
freq = fftpack.fftfreq(len(y))
plt.plot(np.abs(freq), power)
plt.show()

# maintenant on filtre
fourier[power < 200] = 0
plt.plot(np.abs(freq), np.abs(fourier))
plt.show()

# on applique la transformée inverse
filtered_signal = fftpack.ifft(fourier)
plt.figure(figsize=(12, 8))
plt.plot(x, y,lw=0.5)
plt.plot(x, filtered_signal,lw=3)
plt.show()


