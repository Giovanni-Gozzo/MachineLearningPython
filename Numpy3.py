import numpy as np

## mathematique avec numpy

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

Somme = np.sum(A)
print(Somme)

SommeLigne = np.sum(A, axis=1)
print(SommeLigne)
# ca donne la somme des éléments de chaque ligne

SommeColonne = np.sum(A, axis=0)
print(SommeColonne)
# ca donne la somme des éléments de chaque colonne

Sommecumulée = np.cumsum(A)
print(Sommecumulée)
# ca donne la somme cumulée des éléments de chaque ligne

SommecumuléeLigne = np.cumsum(A, axis=1)
print(SommecumuléeLigne)
# ca donne la somme cumulée des éléments de chaque ligne

min = np.min(A)
print(min)

minLigne = np.min(A, axis=1)
print(minLigne)

indexminLigne = np.argmin(A, axis=1)
print(indexminLigne)

indexmin = np.argmin(A)

##Sort

# argsort renvoie les indices qui trieraient le tableau
# sort renvoie le tableau trié
tableautrié = np.sort(A)
print(tableautrié)

indextrié = np.argsort(A)
print(indextrié)

# on peut aussi calcule dans les tableau

Exp = np.exp(A)
print(Exp)

Log = np.log(A)
print(Log)

cos = np.cos(A)
print(cos)

# on peut aussi faire des opérations dans le tableau

moyenne = np.mean(A)
print(moyenne)

moyenneLigne = np.mean(A, axis=1)
print(moyenneLigne)

ecarttype = np.std(A)
print(ecarttype)

variation = np.var(A)
print(variation)

coefficientcorrelation = np.corrcoef(A)
print(coefficientcorrelation)
## ce tableau est une matrice de corrélation qui donne la corrélation entre chaque colonne et chaque ligne

nbfois = np.unique(A, return_counts=True)
print(nbfois)
##ca renvoie les valeurs uniques et le nombre de fois qu'elles apparaissent sousforme de tuple

values, counts = np.unique(A, return_counts=True)
print(values)

print(values[counts.argsort()])
# ca renvoie les valeurs uniques triées par ordre croissant de nombre d'apparition

for i, j in zip(values[counts.argsort()], counts[counts.argsort()]):
    print("la valeur", i, "apparait", j, "fois")

##Nan Correction de données manquantes

# il est possible d'ignorer les données manquantes avec certaines fonctions

moyennans = np.nanmean(A)
print(moyennans)
# les données manquante sont alors juste ignorées

isnans = np.isnan(A)
print(isnans)
# ca renvoie un tableau de booléen qui indique si la valeur est nan ou pas (nan signifie not a number)

nombredeNan = np.sum(isnans)

moynan = np.isnan(A).sum() / A.size
print(moynan)
# ca donne le pourcentage de nan dans le tableau

# on peut aussi remplacer les nan par une valeur
A[isnans] = 0
print(A)
# ca remplace les nan par 0

##Algebre lineaire

A = np.ones((2, 3))
B = np.ones((3, 2))

Transpose = A.T
print(Transpose)
# permet de transposer un tableau (changer les lignes et les colonnes)

ProduitMatriciel = np.dot(A, B)
print(ProduitMatriciel)
ProduitMatriciel2 = np.dot(B, A)
print(ProduitMatriciel2)

# np.linalg.det(A)
# ca permet de calculer le déterminant d'une matrice

# np.linalg.inv(A)
# ca permet de calculer l'inverse d'une matrice

# vecteur propre et valeur propre

# valeurspropres,vecteurspropres=np.linalg.eig(A)


# Exercice
np.random.seed(0)
A = np.random.randint(0, 100, [10, 5])

D=(A-A.mean(axis=0))/A.std(axis=0)

print(D)
