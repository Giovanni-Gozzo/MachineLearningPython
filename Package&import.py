from Dictionnaire import trier,classeur
from Dictionnaire import *

trier(classeur, 5)
print(classeur)

import math
import random
import os
import statistics
import glob

#Math

print(math.cos(2*math.pi))
print(math.exp(10))
print(math.log10(1000))

#statistics
list_notes = [10, 12, 15, 12, 11, 9, 16, 12, 14, 17, 16, 16, 18, 16, 16, 15, 16, 16, 17, 16, 16, 16, 16, 16, 16, 16, 16]
print(statistics.mean(list_notes))
print(statistics.variance(list_notes))

#random
lst_nb = [random.randint(0, 100) for i in range(20)]
print(random.choice(lst_nb))

random.random()
#cela renvoie un nombre aléatoire entre 0 et 1
random.randint(5, 100)
random.randrange(100)

random.sample(lst_nb, 5)
#renvoie 5 nombres aléatoires de la liste lst_nb
random.shuffle(lst_nb)

#os
os.getcwd()
#renvoie le chemin du dossier courant

#glob
glob.glob("*.py")
#renvoie tous les fichiers python du dossier courant


