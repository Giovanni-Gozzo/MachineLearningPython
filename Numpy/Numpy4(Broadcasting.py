import numpy as np

A=np.array([[1,2,3],[4,5,6]])
B=np.array([[9,8,7],[6,5,4]])

print(A*B)
print(A+B)
# il faut que les deux matrices aient la meme taille pour pouvoir les additionner ou les multiplier

# mais le broadcasting permet de faire des operations entre des matrices de tailles differentes
# exemple:

A=np.array([[1,2,3],[4,5,6]])

print(A+100)
# on ajoute 100 a chaque element de la matrice A car le broadcasting permet de faire cette operation et etend le 100
# le broadcasting marche seulement sur les matrices de taille 1
# exemple:

A=np.array([[1,2,3],[4,5,6]])
B=np.array([100,200,300])
print(A+B)
# le broadcasting permet de faire l'addition entre les deux matrices car elles ont la meme taille et ca etend la matrice B
