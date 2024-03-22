##Numpy NArray
import numpy as np

##attribut shape est très utilisé pour connaitre la taille d'un tableau

A=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A.shape)
print(A.size)

B=np.zeros((3,3))

C=np.ones((3,4))

D=np.full((3,3),5)

E= np.random.randn(3,4)
print (E)

D= np.eye(3)
print(D)
# ca donne une matrice identité

F=np.linspace(0,10,5)
# ca donne un tableau de 5 valeurs entre 0 et 10
print(F)

G=np.arange(0,10,2)
# ca donne un tableau de 5 valeurs entre 0 et 10
print(G)

#dtype est très important pour connaitre le type de données par expl: int32, int64, float32, float64
#Cela peut etre précisé lors de la création du tableau par expl:
#H=np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float64)
#ou bien
#H=np.linspace(0,10,5, dtype=np.int32)
#Cela peut joué sur la performance de l'application
#très souvent on laisse le type par défaut
# mais si on veyt optimiser la performance on peut le préciser


#Manipulation de tableau

Atest=np.ones((3,2))
Btest=np.zeros((3,2))

#On peut aussi concatener des tableaux
#Attention il faut que les dimensions soient compatibles
Horizontal2=np.concatenate((Atest,Btest), axis=1)
print(Horizontal2)
Vertical2=np.concatenate((Atest,Btest), axis=0)
print(Vertical2)


#onr peut reshape un tableau
#Attention il faut que les dimensions soient compatibles
Areshape= np.full((3,4),5)

Breshape=Areshape.reshape((2,6))

print(Breshape)

#Reshape très imortant dans calcul matriciel
#Par exmple :

Exemplimp= np.array([1,2,3])
#le shape est (3,)
print(Exemplimp.shape)
#on aimerait bien avoir (3,1) pour pouvoir faire des calculs matriciels
#on peut utiliser reshape
Exemplimp=Exemplimp.reshape((3,1))
print(Exemplimp.shape)

# ou inversement
Exemplimp=Exemplimp.squeeze()
print(Exemplimp.shape)

#methode ravel
#permet de transformer un tableau en un vecteur

Exemplimp=Exemplimp.ravel()
#transforme en 1D


#Exercice
def initialisation(m,n):
    # m : nombre de Lignes
    #n : nombre de colonnes
    # retourne une matrice aléatoire (m, n+1)
    # avec une colonne biais (remplie de "1") tout a droite
    Matrice= np.random.randn(m,n)
    colonne1=np.ones((m,1))
    m=np.concatenate((Matrice,colonne1),axis=1)
    return m

