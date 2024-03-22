# 1) Créer un Train set et un Test set. Entrainer puis évaluer
# 2) Avec GridSearch, trouver Les meilleurs hyper-parametres n_neighbors, metrics et weights
# 3) Est-ce-que collecter plus de données serait utile ?
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier

titanic = sns.load_dataset('titanic')
titanic= titanic[['survived','pclass','sex','age']]
titanic.dropna(axis=0, inplace=True)
titanic['sex'].replace(['male','female'],[0,1],inplace=True)

X=titanic[['pclass','sex','age']].values
Y=titanic['survived'].values

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2)

param_grid={'n_neighbors':np.arange(1,50),'metric':['euclidean','manhattan']}

grid= GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
grid.fit(X_train,Y_train)
print(grid.best_params_)

model=grid.best_estimator_
print(model.score(X_test,Y_test))

erreurmatrix=confusion_matrix(Y_test,model.predict(X_test))
print(erreurmatrix)

train_sizes, train_scores, test_scores = learning_curve(model,X_train,Y_train,train_sizes=np.linspace(0.1,1.0,5),cv=5)

plt.plot(train_sizes,train_scores.mean(axis=1),label='Train')
plt.plot(train_sizes,test_scores.mean(axis=1),label='Test')
plt.xlabel('Train Sizes')
plt.legend()
plt.show()

