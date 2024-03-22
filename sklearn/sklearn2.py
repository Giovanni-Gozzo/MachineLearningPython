import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target
plt.scatter(X[:,0],X[:,1],c=y,alpha=0.8)
plt.show()

from sklearn.model_selection import train_test_split, validation_curve

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#test_size=0.2 signifies 20% of the data is used for testing and 80% for training

print(X_train.shape)
print(X_test.shape)

plt.figure(figsize=(12,6))
plt.subplot(121)
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,alpha=0.8)
plt.title('Train Data')
plt.subplot(122)
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,alpha=0.8)
plt.title('Test Data')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

from sklearn.model_selection import cross_val_score

cross_val_score(KNeighborsClassifier(2),X_train,y_train,cv=5,scoring='accuracy').mean()
#au lieu de test tout les voisins on va utilise valdiation croisee pour trouver le meilleur nombre de voisins

model=KNeighborsClassifier()
k_range=range(1,50)

train_score, test_score = validation_curve(
    model, X_train, y_train, param_name="n_neighbors", param_range=k_range, cv=5)
#cv est le nombre de decoupage de la base de donnee pour la validation croisee pour etre sur que le modele est robuste

plt.plot(k_range,test_score.mean(axis=1))
plt.show()

#malheureusement on ne peut pas utiliser validation_curve pour trouver le meilleur nombre de voisins
#car il ne prend pas en compte les donnees de test
#on va donc utiliser GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors':np.arange(1,20),'metric':['euclidean','manhattan']}
grid= GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
grid.fit(X_train,y_train)
print(grid.best_params_)
model=grid.best_estimator_
print(model.score(X_test,y_test))

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,model.predict(X_test))
#c'est une matrice qui nous permet de voir les erreurs de classification
#la diagonale principale represente les bonnes classifications
#les autres cases representent les erreurs de classification

#en donnant plus de donnees au modele on peut ameliorer la performance ?
# on va regarde ca avec les courbes d'apprentissage

from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(model,X_train,y_train,train_sizes=np.linspace(0.1,1.0,5),cv=5)

plt.plot(train_sizes,train_scores.mean(axis=1),label='Train')
plt.plot(train_sizes,test_scores.mean(axis=1),label='Test')
plt.xlabel('Train Sizes')
plt.legend()
plt.show()
