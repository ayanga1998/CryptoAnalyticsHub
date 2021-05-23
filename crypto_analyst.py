from time import sleep
import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime
import numpy as np
from keras.layers import GRU, Dropout, Dense
from keras import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler



class CryptoAnalyst(object):
    def __init__(self, coin='bitcoin'):
        self.coin = coin
        self.cg = CoinGeckoAPI()

    def get_names_dict(self):
        sleep(1)
        coin_dict = {}
        coins = self.cg.get_coins_markets(vs_currency='usd')
        for coin in coins:
            coin_dict[coin['name']] = coin['id']
        return coin_dict

    def get_coin_names(self):
        list_of_names = self.get_names_dict()
        return list(list_of_names.keys())

    def get_id_from_name(self, value):
        self.coin = self.get_names_dict()[value]
        return self.coin

    def convert_unix(self, time):
        time = int(time) / 1000
        return datetime.utcfromtimestamp(time)

    def get_OHLC(self, days=30):
        sleep(2)
        ohlc = self.cg.get_coin_ohlc_by_id(id=str(self.coin), vs_currency='usd', days=days)
        ohlc_df = pd.DataFrame(ohlc, columns=['date', 'open', 'high', 'low', 'close'])
        ohlc_df['date'] = ohlc_df['date'].apply(lambda x: self.convert_unix(x))
        ohlc_df = ohlc_df.set_index('date')
        return ohlc_df


    def get_historical_price_data(self, days='max'):
        sleep(2)
        market_data = self.cg.get_coin_market_chart_by_id(id = self.coin, vs_currency='usd', days=days)
        market_dict = {'date': [self.convert_unix(data[0]) for data in market_data['prices']],
                       'price': [data[1] for data in market_data['prices']],
                       'volume': [data[1] for data in market_data['total_volumes']],
                       'market_cap': [data[1] for data in market_data['market_caps']]
                       }
        df = pd.DataFrame(market_dict)
        df.set_index('date', inplace = True)
        return df

    def get_historical_price_change(self, day = 'max'):
        price = self.get_historical_price_data()
        date = price.index
        price = price['price'].values
        change = [price[i+1] - price[i] for i in range(0, len(price)-1)]
        return [date[1:], change]

    @property
    def coin(self):
        return self._coin

    @coin.setter
    def coin(self, value):
        print('Setting value...')
        self._coin = value


class CustomModel(CryptoAnalyst):

    def get_data(self):
        sleep(2)
        data = self.get_historical_price_data()
        return data

    def clean_data(self, data):
        df = data.drop(columns=['volume', 'market_cap'])
        df['price_shifted'] = df['price'].shift(-1, fill_value=0)
        df.drop(data.tail(1).index, inplace = True)
        return df

    def train_test_split(self, data, test_size = 0.2):
        train_size = int(len(data)*(1-test_size))
        train_dataset, test_dataset = data.iloc[:train_size], data.iloc[train_size:]

        X_train = train_dataset.drop('price_shifted', axis=1)
        y_train = train_dataset.loc[:, ['price_shifted']]

        X_test = test_dataset.drop('price_shifted', axis = 1)
        y_test = test_dataset.loc[:, ['price_shifted']]
        return X_train, y_train,  X_test, y_test

    def data_preprocess(self, xtrain, ytrain, xtest, ytest, scaler_x, scaler_y):
        input_scaler = scaler_x.fit(xtrain)
        output_scaler = scaler_y.fit(ytrain)

        train_y_norm = output_scaler.transform(ytrain)
        train_x_norm = input_scaler.transform(xtrain)

        test_y_norm = output_scaler.transform(ytest)
        test_x_norm = input_scaler.transform(xtest)

        return train_x_norm, train_y_norm, test_x_norm, test_y_norm

    def threeD_dataset(self, X, y, time_steps = 1):
        Xs, ys = [], []

        for i in range(len(X) - time_steps):
            v = X[i:i+time_steps, :]
            Xs.append(v)
            ys.append(y[i+time_steps])

        return np.array(Xs), np.array(ys)

    def create_model(self, xtrain):
        model = Sequential()
        model.add(GRU(units = 64, return_sequences = True, input_shape = [xtrain.shape[1], xtrain.shape[2]]))
        model.add(Dropout(0.2))
        model.add(GRU(units = 64))
        model.add(Dropout(0.2))
        model.add(Dense(units =1))
        model.compile(loss = 'mse', optimizer = 'adam')
        return model

    def fit_model(self, model, xtrain, ytrain):
        early_stop = EarlyStopping(monitor = 'val_loss', patience = 10)
        history = model.fit(xtrain, ytrain, epochs = 100, validation_split=0.2, batch_size=32, shuffle=False, callbacks = [early_stop])

        return history

    def prediction(self, model, xtest, scaler_y):
        pred = model.predict(xtest)
        pred = scaler_y.inverse_transform(pred)
        return pred

    def run_model(self):
        time_step = 10

        data = self.get_data()
        cleaned_data = self.clean_data(data)
        X_train, y_train, X_test, y_test = self.train_test_split(cleaned_data)

        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

        train_x_norm, train_y_norm, test_x_norm, test_y_norm = self.data_preprocess(X_train, y_train, X_test, y_test, scaler_x, scaler_y)

        X_test, y_test = self.threeD_dataset(test_x_norm, test_y_norm, time_step)
        X_train, y_train = self.threeD_dataset(train_x_norm, train_y_norm, time_step)
        print(X_train.shape)
        model_gru = self.create_model(X_train)
        print('test passed')
        hitory_gru = self.fit_model(model_gru, X_train, y_train)

        y_test = scaler_y.inverse_transform(y_test)
        y_train = scaler_y.inverse_transform(y_train)

        prediction_gru = self.prediction(model_gru, X_test, scaler_y)
        return prediction_gru, y_test








