import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target

plt.scatter(x[:,0],x[:,1],c=y,alpha=0.8)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit, StratifiedKFold, GroupKFold

cv= KFold(5)
print(cross_val_score(KNeighborsClassifier(),x,y,cv=cv))
# Ici on va donc teste la validation du model en séparant le packet d'entrainement en 5 etapes

cv=LeaveOneOut()
print(cross_val_score(KNeighborsClassifier(),x,y,cv=cv))
# Ici on va donc teste la validation du model avec 1 seule carte et en l'entrainant avec le reste donc soit c'est validé soit ca ne l'ai pas
#donc soit 1 ou soit 0 alors que avant c'est un pourcenateg de reussite
# Attention cela consomme enormement car trop de test de validation

cv=ShuffleSplit(4, test_size=0.2)
print(cross_val_score(KNeighborsClassifier(),x,y,cv=cv))

# Sufflesplut ressemble au KFold mais melange avant de separé a chaque fois et permet de déterminé une size de test aussi
# mais encore une fois probleme on peut se retrouver a entraine une partie qui ne comporte pas de fleur d'iris car très peu d'echantillon
# mais dans la validation il y en a donc il trouvera pas c'est pourquoi on va voir le stratified Kfold qui permet de séparé chaque catégorie
#avant d'entraine ou de validé

cv=StratifiedKFold(4)
print(cross_val_score(KNeighborsClassifier(),x,y,cv=cv))

#Group KFold est très très important et très puissant car il divise par groupe d'invidus donc si on prend un jeu de carte se sera par couleur
#On peut utiliser aussi Groupe shuffle split ce qui ets très bien aussi
cv=GroupKFold(5).get_n_splits(x,y,groups=X[:,0])
#exemple pour le titanic tu peux mettre les différente pclass dans le groups
print(cross_val_score(KNeighborsClassifier(),x,y,cv=cv))




