import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class ForecastTrainer:
    def __init__(self, train_df, test_df):
        self.train = train_df
        self.test = test_df
        self.results = {}
        self.models = {}

    def run_sarima(self):
        """Mandatory Model 1: SARIMA"""
        # Simplified: training on the global trend (can be looped per state)
        model = SARIMAX(self.train['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        model_fit = model.fit(disp=False)
        preds = model_fit.forecast(len(self.test))
        self.results['SARIMA'] = mean_absolute_percentage_error(self.test['Sales'], preds)
        self.models['SARIMA'] = model_fit

    def run_prophet(self):
        """Mandatory Model 2: Facebook Prophet"""
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        # Prophet requires specific column names
        train_p = self.train[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
        m.fit(train_p)
        future = self.test[['Date']].rename(columns={'Date': 'ds'})
        forecast = m.predict(future)
        self.results['Prophet'] = mean_absolute_percentage_error(self.test['Sales'], forecast['yhat'])
        self.models['Prophet'] = m

    def run_xgboost(self):
        """Mandatory Model 3: XGBoost with Lags"""
        features = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_4', 'month', 'day_of_week']
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(self.train[features], self.train['Sales'])
        preds = model.predict(self.test[features])
        self.results['XGBoost'] = mean_absolute_percentage_error(self.test['Sales'], preds)
        self.models['XGBoost'] = model

    def run_lstm(self):
        """Mandatory Model 4: Deep Learning LSTM"""
        # Scaling is mandatory for LSTM
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(self.train[['Sales']])
        
        # Reshape to (samples, time_steps, features)
        X_train = train_scaled.reshape((train_scaled.shape[0], 1, 1))
        y_train = train_scaled
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, verbose=0)
        
        # Predict
        test_scaled = scaler.transform(self.test[['Sales']])
        X_test = test_scaled.reshape((test_scaled.shape[0], 1, 1))
        preds_scaled = model.predict(X_test)
        preds = scaler.inverse_transform(preds_scaled)
        
        self.results['LSTM'] = mean_absolute_percentage_error(self.test['Sales'], preds)
        self.models['LSTM'] = (model, scaler)

    def select_best_model(self):
        """Compares scores and saves the winner."""
        self.run_sarima()
        self.run_prophet()
        self.run_xgboost()
        self.run_lstm()
        
        best_name = min(self.results, key=self.results.get)
        print(f"Tournament Results: {self.results}")
        print(f"🏆 Winner: {best_name}")
        
        # Save the winner
        joblib.dump(self.models[best_name], "models/best_model.pkl")
        return best_name