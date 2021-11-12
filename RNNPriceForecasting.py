import numpy as np
import datetime
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Collect data from Yahoo Finance for the past 1825 days (~ 5 years)
stock = 'BTC-USD'
filepath = 'database/' + stock + '-RNN' '.csv'
name = f'{stock}-RNN-PRED-{date.today()}'
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
training_data_array = scaler.fit_transform(training_data)

# pandas.DataFrame.shape returns a tuple repr the dimensionality of the DataFrame
X_train, Y_train = [], []

# Create 60 training sets: X_train contains data, Y_train contains open price 
training_length = 60
for i in range(training_length, training_data_array.shape[0]):
    X_train.append(training_data_array[i - training_length : i])
    Y_train.append(training_data_array[i,2])

X_train, Y_train = np.array(X_train), np.array(Y_train)

# Predict stock price using Long Short-Term Memory (LSTM) neural network
# RNN Initialization
model = Sequential() 

# LSTM layer 1
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
model.add(Dropout(0.2))

# LSTM layer 2,3,4
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4)) 

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5)) 

# Fully connected layer
model.add(Dense(units =1))
model.summary()

EPOCHS = 20
BATCH_SIZE = 50

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN model
history = model.fit(X_train, Y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_split=0.1)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, color = 'blue', label = 'Training loss')
plt.plot(epochs, val_loss, color = 'red', label = 'Validation loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test Dataset
part_60_days = training_data.tail(60)
df = part_60_days.append(test_data, ignore_index = True)
df = df.drop(['Date', 'Adj Close'], axis = 1)

inputs = scaler.transform(df)
X_test, Y_test = [], []

for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i]) 
    Y_test.append(inputs[i, 0])

X_test, Y_test = np.array(X_test), np.array(Y_test) 
Y_pred = model.predict(X_test)

scale = 1/scaler.scale_[0]
Y_test = Y_test*scale
Y_pred = Y_pred*scale

plt.figure(figsize=(14,5))
plt.plot(Y_test, color = 'red', label = 'Real Bitcoin Price')
plt.plot(Y_pred, color = 'green', label = 'Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction using RNN-LSTM')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.savefig(f"plots/{name}-{EPOCHS}-epochs.png")
plt.show()




