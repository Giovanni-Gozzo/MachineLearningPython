import pandas as pd

df= pd.read_excel("../titanic.xls")
df.drop(['name', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1,inplace=True)

df.loc[df['age'] < 20, 'category'] = 'cat0'
df.loc[(df['age'] > 20) & (df['age'] < 30), 'category'] = 'cat1'
df.loc[(df['age'] > 30) & (df['age'] < 40), 'category'] = 'cat2'
df.loc[df['age'] > 40, 'category'] = 'cat3'

print(df)
