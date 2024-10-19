import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd


class Model(nn.Module):
    # 4 in features parce qu'il y a la taille des petal la largeur ... il y en a 4
    # 8 h1 et 9 h2 sont des hyperparametres qui sont des nombres de neurones
    # 3 out features parce qu'il y a 3 types de fleurs
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


model = Model()

url = '../iris.csv'
df = pd.read_csv(url)

print(df.head())

df['species'] = df['species'].replace('setosa', 0.0)
df['species'] = df['species'].replace('versicolor', 1.0)
df['species'] = df['species'].replace('virginica', 2.0)

# Train Test Split

X = df.drop('species', axis=1).values
y = df['species'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Measure Erreur
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model.parameters)

#train model
epochs = 200
loses = []

for i in range(epochs):
    y_pred= model.forward(X_train)

    loss= criterion(y_pred, y_train)
    loses.append(loss.detach().numpy())

    if i % 10 == 0:
        print(f'epoch {i} and loss is: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), loses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()


# Evaluate the model
with torch.no_grad():
    y_eval=model.forward(X_test)
    loss= criterion(y_eval, y_test)
    print(loss)

#L'evaluation avec les données de test sont assez eloigné dela perte que nous avions obtenu avec les données d'entrainement
#donc on va essayer de trouver le probleme

correct=0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val= model.forward(data)

        print(f'{i+1}. {str(y_val)} {y_test[i]}'+str(y_val.argmax().item()))

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'We got {correct} correct!')

# Tester avec des nouvelles données

mystery_iris = torch.tensor([4.7,3.2, 1.3, 0.2])

with torch.no_grad():
    print(model(mystery_iris))

new_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])

with torch.no_grad():
    print(model(new_iris))

# Save the model
torch.save(model.state_dict(), 'my_iris_model.pt')

# Load the model
new_model = Model()
new_model.load_state_dict(torch.load('my_iris_model.pt'))

print(new_model.eval())
