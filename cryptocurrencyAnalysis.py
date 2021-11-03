# towardsdatascience.com : Generating trade signals using MA crossover strategy
# sanjoshaju github: cryptocurrenct-analysis-python

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Save crypto data into a csv file

start_date = '2021-10-01'
end_date = '2021-11-03'

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

df = pd.DataFrame({'Date' : btc_df['Date'] , 'BTC': btc_df['Close'], 'ETH': eth_df['Close'], 'BNB' : bnb_df['Close'], 'SHIB' : shib_df['Close']})


print(df)
df.plot(x = 'Date' ,figsize = (15,8))
plt.grid()
plt.ylabel("Price in USD")
plt.show()





    