# BuildingAI


This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 

##  Stock Analysis with Python

This Python script provides a comprehensive analysis of stock data using various libraries like `pandas`, `numpy`, `matplotlib`, and `seaborn`. It focuses on four major tech companies - Apple, Google, Microsoft, and Amazon. The script covers data extraction, visualization, moving averages, daily returns, correlation analysis, and stock price prediction using a LSTM neural network model.


## Summary

This script is a comprehensive toolkit for stock data analysis and prediction. It covers from basic data visualization to advanced predictive modeling using deep learning. The modular structure allows for easy adaptation to different stocks or financial metrics.

### Libraries and Setup

- **Pandas & Numpy:** For data manipulation and numerical calculations.
- **Matplotlib & Seaborn:** For data visualization.
- **pandas_datareader & yfinance:** To fetch financial data from Yahoo Finance.
- **datetime:** To work with date and time objects.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
```

### Initial Configuration

- Setting up plot styles and overriding the pandas datareader to use `yfinance` instead.
- Defining the list of tech companies and setting the time frame for data analysis.

```python
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
yf.pdr_override()

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
```

### Data Acquisition

- Fetching historical stock data for each company and concatenating them into a single DataFrame.
- Adding company names to the DataFrame.

```python
for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)

company_list = [AAPL, GOOG, MSFT, AMZN]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)
```

### Data Visualization and Analysis

- Visualizing adjusted closing prices and volumes traded.
- Calculating and plotting moving averages (10, 20, 50 days).
- Analyzing daily returns and plotting histograms.
- Creating correlation matrices between the stocks.

```python
# Example: Plotting Adjusted Close Price
plt.figure(figsize=(15, 10))
for i, company in enumerate(company_list, 1):
    plt.subplot(2, 2, i)
    company['Adj Close'].plot()
    plt.title(f"Closing Price of {tech_list[i - 1]}")
plt.tight_layout()
```

### Predictive Analysis Using LSTM

- Preprocessing the data for the LSTM model (scaling, creating training and test datasets).
- Building and training the LSTM model on the closing price of Apple stock.
- Predicting future stock prices and calculating the model's performance (RMSE).

```python
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Example: Building the LSTM Model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)
```

### Final Visualization

- Visualizing actual vs predicted prices of Apple stock using matplotlib.

```python
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
```

