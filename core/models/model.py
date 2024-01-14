import datetime as dt
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
import yfinance as yf


class Model:

    MODEL_DIRECTORY = 'trained_models'
    TRAIN_DATES = {
        "start": dt.datetime(2012, 1, 1),
        "end": dt.datetime(2023, 1, 1)
    }
    PREDICT_DATES = {
        "start": dt.datetime(2023, 1, 1),
        "end": dt.datetime.now()
    }
    PREDICTION_DAYS = 60
    SCALER = MinMaxScaler(feature_range=(0, 1))

    def __init__(self, company):
        self.company = company
        self.model = None

    @property
    def _model(self):
        return self.model or self._load() or self._train()

    def _load(self):
        try:
            self.model = load_model(pathlib.PurePath(self.MODEL_DIRECTORY, self.company))
        except OSError:
            return None
        return self.model

    def _train(self):
        scaled_data = self._prepare_data(**self.TRAIN_DATES)
        x_train = []
        y_train = []
        for x in range(self.PREDICTION_DAYS, len(scaled_data)):
            x_train.append(scaled_data[x - self.PREDICTION_DAYS: x])
            y_train.append(scaled_data[x])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=25, batch_size=32)
        model.save(pathlib.PurePath(self.MODEL_DIRECTORY, self.company))
        return model

    def predict(self, days_in_future=5):
        model_input = self._prepare_data(**self.PREDICT_DATES)
        future_predictions = []
        for _ in range(days_in_future):
            real_data = [model_input[len(model_input) - self.PREDICTION_DAYS: len(model_input)]]
            real_data = np.array(real_data)
            real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

            prediction = self._model.predict(real_data)
            future_predictions.append(self.SCALER.inverse_transform(prediction)[0, 0])

            # Update model_input to include the predicted value for the current day
            model_input = np.append(model_input, prediction)
        return future_predictions

    def plot(self, future_predictions_days=15):
        data = yf.download(self.company, **self.TRAIN_DATES)
        test_data = yf.download(self.company, **self.PREDICT_DATES)
        actual_prices = test_data['Close'].values
        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

        model_input = total_dataset[len(total_dataset) - len(test_data) - self.PREDICTION_DAYS:].values
        model_input = model_input.reshape(-1, 1)
        model_input = self.SCALER.transform(model_input)

        x_test = [model_input[x - self.PREDICTION_DAYS:x, 0] for x in range(self.PREDICTION_DAYS, len(model_input))]

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_prices = self._model.predict(x_test)
        p = self.SCALER.inverse_transform(predicted_prices)

        plt.plot(actual_prices, color="black", label='actual price')
        plt.plot(p, color="green", label='predicted price')

        future_predictions = self.predict(future_predictions_days)

        shift = len(actual_prices) #274
        size_future = len(future_predictions) #15
        x = np.arange(size_future) + shift
        plt.plot(x, future_predictions, color="red", label="future predictions")

        plt.xlabel('Time')
        plt.ylabel(f"{self.company} share price")
        plt.legend()
        plt.show()

    def _prepare_data(self, start, end):
        try:
            data = yf.download(self.company, start=start, end=end)
            scaled_data = self.SCALER.fit_transform(data['Close'].values.reshape(-1, 1)).flatten()
        except ValueError:
            print(f'Stock {self.company} is not listed')
            exit(0)
        return scaled_data
