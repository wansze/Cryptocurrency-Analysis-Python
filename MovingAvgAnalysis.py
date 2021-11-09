import datetime
from datetime import date
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt

# Using Moving Averages as Price Signals
# When Short MA crosses above long MA: Bullish Trading Signal (buying point)
# When Short MA falls below long MA: Bearish Trading Signal (selling point)

# Function for getting stock price data and storing it in a dataframe
def stockpriceanalysis(stock):

    # Grab financial data from Yahoo Finance from the past 1200 days
    end_date = date.today()
    start_date = end_date - datetime.timedelta(days = 1200)
    stock_df = pdr.DataReader(stock, 'yahoo', start_date, end_date)

    # Convert from dict to DataFrame
    stock_df = pd.DataFrame.from_dict(stock_df)

    # Calculate the Simple Moving Average using 20 days and 250 days
    # Short Term Average: 20 days (~ trading days in a month)
    # Long Term Average: 250 days (~ trading days in a year)
    stock_df['20days'] = stock_df['Close'].rolling(20).mean()
    stock_df['250days'] = stock_df['Close'].rolling(250).mean()

    stock_df[['Close','20days','250days']].plot(figsize = (10,4))
    plt.grid(True)
    plt.title(stock + " Price Moving Averages")
    plt.axis('tight')
    plt.ylabel('Price')
    plt.show()
    

# Example: Pass Bitcoin stock ticker as an argument for testing function 
stockpriceanalysis('BTC-USD')

