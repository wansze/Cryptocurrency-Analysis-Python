import numpy as np
import datetime
from datetime import date
import matplotlib as plt
import pandas as pd
import pandas_datareader as pdr
from sklearn import preprocessing

# Collect data from Yahoo Finance for the past 1825 days (~ 5 years)
stock = 'BTC-USD'
filepath = 'database/' + stock + '-RNN' '.csv'
end_date = date.today()
start_date = end_date - datetime.timedelta(days = 1825)
pdr.DataReader(stock, 'yahoo', start_date, end_date).to_csv(filepath)

# Store data in a dataframe
btc_df = pd.read_csv(filepath)
btc_df['Date'] = pd.to_datetime(btc_df['Date'], format='%Y-%m-%d')

# Copy data for training the RNN (data before the past year)
training_data = btc_df[btc_df['Date'].dt.date < end_date - datetime.timedelta(days = 365)].copy()

# Copy data for testing the RNN (data from the past year)
test_data = btc_df[btc_df['Date'].dt.date > end_date - datetime.timedelta(days = 365)].copy()

# Drop date and adj_close columns
training_data = training_data.drop(['Date', 'Adj Close'], axis = 1)

# Normalize the data values
scaler = preprocessing.MinMaxScaler()
training_data = scaler.fit_transform(training_data)
print(training_data)


