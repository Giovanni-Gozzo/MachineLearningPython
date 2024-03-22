## Quelques fonctions

x=-3
print(abs(x))
## abs() renvoie la valeur absolue d'un nombre

x=3.14
print(round(x))
## round() arrondi un nombre à l'entier le plus proche

liste_1=[1,2,3,4,5,6,7,8,9,10]
print(sum(liste_1))
## sum() renvoie la somme des éléments d'une liste

liste_2=[1,2,3,4,5,6,7,8,9,10]
print(min(liste_2))
## min() renvoie la valeur minimale d'une liste

liste_3=[1,2,3,4,5,6,7,8,9,10]
print(max(liste_3))
## max() renvoie la valeur maximale d'une liste

liste_4=[1,2,3,4,5,6,7,8,9,10]
print(len(liste_4))
## len() renvoie le nombre d'éléments d'une liste

liste_bool=[True, False, True, True]
print(all(liste_bool))
## all() renvoie True si tous les éléments d'une liste sont True

liste_bool=[True, False, True, True]
print(any(liste_bool))
## any() renvoie True si au moins un élément d'une liste est True

x=10
print(type(x))
## type() renvoie le type d'un objet

x=str(10)
print(type(x))

y='20'
y=int(y)
print(type(y))
## int() convertit un objet en entier

x=10
x=float(x)

liste_1=[1,2,3,4,5,6,7,8,9,10]
tuple_1=tuple(liste_1)
print(tuple_1)

liste_2=list(tuple_1)
print(liste_2)

inventaire={'pommes': 430, 'bananes': 312, 'oranges': 525, 'poires': 217}
list_fruits=list(inventaire.keys())
print(list_fruits)
list_nb_fruits=list(inventaire.values())
print(list_nb_fruits)

x=25
ville='Paris'
message='La température est de {} degrés à {}'.format(x, ville)
message2=f'La température est de {x} degrés à {ville}'
print(message)

import numpy as np
parametre={
    'W1':np.random.randn(2,2),
    'b1':np.zeros((2,1)),
    'W2':np.random.randn(2,2),
    'b2':np.zeros((2,1)),
}
for i in range(1,3):
    print('couche', i)
    print(parametre['W'+str(i)])

##fonction open
f=open('fichier.txt', 'w')
f.write('Bonjour')
f.close()
f=open('fichier.txt', 'r')
print(f.read())

with open('fichier.txt', 'w') as f:
    for i in range(10):
        f.write("la valeur de {} au carré est {}\n".format(i, i**2))

with open('fichier.txt', 'r') as f:
    print(f.read())





