import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Save crypto data into a csv file
start_date = '2021-01-01'
end_date = '2021-11-06'

cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SHIB-USD'] 

filepaths = []

for crypto in cryptos:
    filepath = 'database/' + crypto + '.csv'
    filepaths.append(filepath)
    pdr.DataReader(crypto, 'yahoo', start_date, end_date).to_csv(filepath)

btc_df = pd.read_csv(filepaths[0])
eth_df = pd.read_csv(filepaths[1])
bnb_df = pd.read_csv(filepaths[2])
shib_df = pd.read_csv(filepaths[3])

df = pd.DataFrame({'BTC': btc_df['Close'], 'ETH': eth_df['Close'], 'BNB' : bnb_df['Close'], 'SHIB' : shib_df['Close']})

# scale the data
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
scaled = min_max_scaler.fit_transform(df)
print(scaled)

# convert scaled numpy array into dataframe and plot it
df_scaled = pd.DataFrame(scaled, columns = df.columns)
df_scaled['Date'] = btc_df['Date']

plt.style.use('fivethirtyeight')
df_scaled.plot(x = 'Date' ,figsize = (15,8), fontsize=11, ylim = [0,100])
plt.xlabel("Days")
plt.ylabel("Scaled Crypto Price ($)", fontdict={'fontsize':11})
plt.title("Crypto scaled graph", fontdict={'fontsize':11})
plt.show()












    