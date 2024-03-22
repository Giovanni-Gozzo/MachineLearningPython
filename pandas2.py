import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bitcoin = pd.read_csv("../Bitcoin EUR history.csv", index_col="Date", parse_dates=True)
print(bitcoin.head())

bitcoin['Close'].plot()
plt.show()

bitcoin['Close']['2023-09':'2023-12'].plot()
plt.show()

# pandas connais les date et peut donc faire des opérations dessus
# si il comprend pas une date il est possible d'utiliser to_datetime jusque milliseconde
print(pd.to_datetime('2023-09-01'))
print(pd.to_datetime('2023-09-01 15:30:00'))
print(pd.to_datetime(['2023-09-01', '2023-09-02']))

# Resample
resampled_data = bitcoin.loc["2019", "Close"].resample('W').mean()
resampled_data.plot()
plt.show()

resampled_data = bitcoin.loc["2019", "Close"].resample('2W').mean()
resampled_data.plot()
plt.show()

# permet de voir la volatilité du bitcoin
resampled_data = bitcoin.loc["2023", "Close"].resample('2W').std()
resampled_data.plot()
plt.show()

plt.figure(figsize=(12, 8))
bitcoin.loc["2023", "Close"].plot()
bitcoin.loc["2023", "Close"].resample('M').mean().plot(label="Moyenne par mois", lw=3, ls=":", alpha=0.8)
bitcoin.loc["2023", "Close"].resample('W').mean().plot(label="Moyenne par semaine", lw=2, ls="--", alpha=0.8)
plt.legend()
plt.show()

m = bitcoin["Close"].resample('W').agg(['mean', 'std', 'min', 'max'])

plt.figure(figsize=(12, 8))
m['mean']['2023'].plot(label="Moyenne par mois")
plt.fill_between(m.index, m['max'], m['min'], alpha=0.2, label="min-max par mois")
plt.legend()
plt.show()

bitcoin.loc["2023", "Close"].rolling(window=7).mean().plot()
bitcoin.loc["2023", "Close"].rolling(window=7).mean().plot()
plt.show()

plt.figure(figsize=(12, 8))
bitcoin.loc["2023", "Close"].plot()
bitcoin.loc["2023", "Close"].rolling(window=7, center=True).mean().plot(label="Moyenne mobile 7 jours")
bitcoin.loc["2023", "Close"].rolling(window=30, center=True).mean().plot(label="Moyenne mobile 30 jours")
plt.legend()
plt.show()

## moyenne mobile exponentielle
plt.figure(figsize=(12, 8))
bitcoin.loc["2023", "Close"].plot()
bitcoin.loc["2023", "Close"].rolling(window=7, center=True).mean().plot(label="Moyenne mobile 7 jours")
bitcoin.loc["2023", "Close"].rolling(window=30, center=True).mean().plot(label="Moyenne mobile 30 jours")
bitcoin.loc["2023", "Close"].ewm(alpha=0.6).mean().plot(label="Moyenne mobile 30 jours")
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
bitcoin.loc["2023", "Close"].plot()
for i in np.arange(0.2, 1, 0.2):
    bitcoin.loc["2023", "Close"].ewm(alpha=i).mean().plot(label="alpha=" + str(i))
plt.legend()
plt.show()

#assemble 2 dataset ensemble

Ether= pd.read_csv("../Ethere.csv", index_col="Date", parse_dates=True)
pd.merge(bitcoin, Ether, on="Date", how="inner", suffixes=('_BTC','_ETH')).head()
# inner = intersection des 2 dataset donc les date en commun
# outer = union des 2 dataset donc les date pas en commun sont rempli par NaN
# left = garde les date du dataset de gauche
# right = garde les date du dataset de droite

btc_eth = pd.merge(bitcoin, Ether, on="Date", how="inner", suffixes=('_BTC','_ETH'))
btc_eth[['Close_BTC', 'Close_ETH']].plot(subplots=True)
plt.show()
#subplots = affiche les graphique sur 2 graphique différent
# car sinon si meme graphique comparer on voit pas bien les différence

#corrélations
print(btc_eth[['Close_BTC', 'Close_ETH']].corr())
# 1 = corrélation 91 pourcent ce qui est enorme



