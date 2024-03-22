# scikit learn permet de faire du machine learning avec different model de regression, classification, clustering, etc.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

np.random.seed(0)
m=100
X = np.linspace(0,10,m).reshape(m,1)
y = X + np.random.randn(m,1)
plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
model.score(X,y)
predictions = model.predict(X)

plt.scatter(X,y)
plt.plot(X,predictions,c='r')
plt.show()

# Classification
titanic = sns.load_dataset('titanic')
titanic= titanic[['survived','pclass','sex','age']]
titanic.dropna(axis=0, inplace=True)
titanic['sex'].replace(['male','female'],[0,1],inplace=True)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
y=titanic['survived']
X=titanic.drop('survived',axis=1)
model.fit(X,y)
model.score(X,y)

def survie(model,pclass=3,sex=0,age=20):
    x=np.array([pclass,sex,age]).reshape(1,3)
    print(model.predict(x))
    print(model.predict_proba(x))

survie(model)

tableau = []
for i in range(1,10):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X,y)
    print(model.score(X,y))
    tableau.append(model.score(X,y))


def index_max(tableau):
    for i in tableau:
        if i == max(tableau):
            return tableau.index(i)


print(index_max(tableau)+1)
