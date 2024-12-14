# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:33:27 2024

@author: Oghale Enwa
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('all_stocks_5yr.csv')
sp = raw_data.copy()
sp['date'] = pd.to_datetime(sp['date'], format='%Y-%m-%d')
#print(sp.isnull().values.any())
#print(sp.isnull().sum())

#missing_values = sp.isnull().sum()
#missing_values_percentage = (missing_values / len(sp)) * 100

# Print the results
#print("Missing Values Count:\n", missing_values)
#print("\nMissing Values Percentage:\n", missing_values_percentage)

data_filled = sp.ffill()

# Check if there are any remaining missing values after the operation
#remaining_missing_values = data_filled.isnull().sum()
#print("Remaining Missing Values:\n", remaining_missing_values)


#plt.figure(figsize=(12, 6))
#plt.plot(data_filled['date'], data_filled['close'], label='Close Price')
#plt.title('Closing Price Over Time')
#plt.xlabel('Date')
#plt.ylabel('Close Price (USD)')
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)

#plt.figure(figsize=(12, 6))
#plt.plot(data_filled['date'], data_filled['open'], label='Open Price')
#plt.title('Opening Price Over Time')
#plt.xlabel('Date')
#plt.ylabel('Open Price (USD)')
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)

#plt.figure(figsize=(12, 6))
#plt.plot(data_filled['date'], data_filled['volume'], label='Trading Volume', color='orange')
#plt.title('Trading Volume Over Time')
#plt.xlabel('Date')
#plt.ylabel('Volume')
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_filled['close'].values.reshape(-1, 1))

# Creating the dataset for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60  # Number of previous days' data used to predict the next day
X, Y = create_dataset(scaled_data, look_back)

# Splitting dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Model evaluation
loss = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")

# Predictions
predicted_stock_price = model.predict(X_test)
# Inverse scaling for a better understanding
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)











