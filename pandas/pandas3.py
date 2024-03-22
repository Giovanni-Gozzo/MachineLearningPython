# https://github.com/mwaskom/seaborn-data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris= pd.read_csv("../iris.csv")
print(iris.head())

plt.scatter(iris['sepal_length'], iris['sepal_width'])
plt.show()
#cela n'est pas très parlant
# il faudrait rajouter des couleurs pour les différentes espèces
# il faudrait rajouter une légende
# il faudrait rajouter des titres
#Bref il faudrait raojuter beaucoup de ligne

#on va donc utiliser seaborn
import seaborn as sns
sns.pairplot(iris, hue='species')
plt.show()

#sns.fonction(x, y, data, hue, size, style)
#x, y : nom des colonnes
#data : dataframe
#hue : colonne pour la couleur
#size : colonne pour la taille
#style : colonne pour le style

titanic= sns.load_dataset('titanic')
print(titanic.head())
titanic.drop(['alone', 'alive', 'who', 'adult_male', 'embark_town', 'class'], axis=1, inplace=True)
titanic.dropna(axis=0,inplace=True)
#enlever les colonnes qui ne servent à rien
sns.pairplot(titanic)
plt.show()
#on voit que les données sont catégorisés et donc illisible sur ce genre de graphoique

#on va donc utiliser la fonction catplot
sns.catplot(x='pclass',y='age',data=titanic,hue='sex')
plt.show()

sns.boxplot(x='pclass',y='age',data=titanic,hue='sex')
plt.show()

#maintenant on veut voir des distributions

sns.distplot(titanic['fare'])
plt.show()
#on voit que la distribution est très asymétrique

sns.jointplot(x='age',y='fare',data=titanic,kind='hex')
plt.show()

#les fonctions les plus utilisées sont
#sns.pairplot()
#sns.catplot()
#sns.boxplot()
#sns.distplot()
#sns.jointplot()
#sns.heatmap()
