import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


# Define the start and end dates for the data
start_date = dt.datetime(2012, 1, 1)
end_date = dt.datetime(2023, 6, 1)

# Fetch Apple Inc. (AAPL) stock price data from Yahoo Finance
# data = pdr.DataReader('AAPL', 'yahoo', start_date, end_date)
data = yf.download('TLSA', start=start_date, end=end_date)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1)).flatten()

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days: x])
    y_train.append(scaled_data[x])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
#
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=25, batch_size=32)
#
# model.save('AAPL')

model = load_model('/home/joe/Code/ductrade/TLSA')

breakpoint()

test_start = dt.datetime(2023, 1, 1)
test_end = dt.datetime.now()

test_data = yf.download('TSLA', start=test_start, end=test_end)

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_input = model_input.reshape(-1, 1)
model_input = scaler.transform(model_input)

x_test = []

for x in range(prediction_days, len(model_input)):
    x_test.append(model_input[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predicted_prices = model.predict(x_test)
p = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color="black", label='actual price')
plt.plot(p, color="green", label='predicted price')

plt.xlabel('Time')
plt.ylabel('share price')

plt.legend()
plt.show()


## Predict next day

# real_data = [model_input[len(model_input) + 1 - prediction_days:len(model_input), 0]]
# real_data = np.array(real_data)
# real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
#
# prediction = model.predict(real_data)
# print(f"prediction : {scaler.inverse_transform(prediction)}")


# Define the number of days you want to predict into the future
# Define the number of days you want to predict into the future
num_days = 10  # Change this to the desired number of days

# Create a list to store the predictions
future_predictions = []

for _ in range(num_days):
    # Predict the next day
    real_data = [model_input[len(model_input) - prediction_days: len(model_input)]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    future_predictions.append(scaler.inverse_transform(prediction)[0, 0])

    # Update model_input to include the predicted value for the current day
    model_input = np.append(model_input, prediction)

# Print the predictions for the next x days
for i, pred in enumerate(future_predictions):
    print(f"Prediction for day {i+1}: {pred}")
