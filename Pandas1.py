import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("../titanic.xls")

# cela nous donne le nombre de lignes et de colonnes
print(df.shape)
# cela nous donne le nom des colonnes
print(df.columns)

# On va supprimer les colonnes qui ne nous intéressent pas
df = df.drop(['name', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1)
print(df.head())

# Cela produit les statistiques de base pour chaque colonne
df.describe()

# grace a ca on peut se rendre compte qu'il nous manque des donnée ici l'age

# 2 options s'offre a nous soit on supprime les lignes qui ont des valeurs manquantes
# soit on remplace les valeurs manquantes par la moyenne de la colonne

# ici on va supprimer les lignes qui ont des valeurs manquantes pour eviter de fausse la réalité

df = df.dropna(axis=0)

# on verifie que l'on a bien supprimer les lignes qui ont des valeurs manquantes
print(df.shape)
# cela va egalement impacter les statistiques de base
print(df.describe())

# Cela nous donne le nombre de passager par classe
print(df['pclass'].value_counts())

# on va commencer  créer notre premier graphique
df['pclass'].value_counts().plot.bar()
plt.show()

df['age'].hist()
plt.show()

moyparsexe = df.groupby(['sex', 'pclass']).mean()
print(moyparsexe)

# les dataframes ont toujours un index

# cela sert a changer l'index
df = df.set_index('name', inplace=True)

#on va voir comment selectionner seulement les passager mineur

df = df[df['age'] < 18]

#pour utiliser les dataframe pandas comme des tableaux numpy
matrice= df.iloc[0:5, 0:2]
print(matrice)

#on peut aussi utiliser les noms des colonnes
matrice= df.loc[0:5, 'pclass':'age']
print(matrice)
#cela va nous donner les 5 premieres lignes et les colonnes pclass a age
