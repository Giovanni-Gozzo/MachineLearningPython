import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bitcoin = pd.read_csv("../Bitcoin EUR history.csv", index_col="Date", parse_dates=True)
bitcoin['buy']=np.zeros(len(bitcoin))
bitcoin['sell']=np.zeros(len(bitcoin))
bitcoin['rollingmax']=bitcoin['Close'].shift(1).rolling(window=28).max()
bitcoin['rollingmin']=bitcoin['Close'].shift(1).rolling(window=28).min()
bitcoin.loc[bitcoin['Close']>bitcoin['rollingmax'],'buy']=1
bitcoin.loc[bitcoin['Close']<bitcoin['rollingmin'],'sell']=1

start=2023
end=2023
fix, ax = plt.subplots(2,figsize=(12,8), sharex=True)
ax[0].plot(bitcoin['Close'][start:end])
ax[0].plot(bitcoin['rollingmax'][start:end])
ax[0].plot(bitcoin['rollingmin'][start:end])
ax[0].legend(['Close','rollingmax','rollingmin'])
ax[1].plot(bitcoin['buy'][start:end], color='g')
ax[1].plot(bitcoin['sell'][start:end], color='r')
ax[1].legend(['buy','sell'])
plt.show()


