import joblib
import pandas as pd
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

class ForecastTrainer:
    def __init__(self, train_data, test_data):
        self.train = train_data
        self.test = test_data
        self.best_model = None
        self.best_score = float('inf')
        self.best_name = ""

    def run_all(self):
        # 1. Simple XGBoost Model
        X_train = self.train[['lag_1', 'lag_7', 'lag_30', 'is_holiday']]
        y_train = self.train['Sales']
        X_test = self.test[['lag_1', 'lag_7', 'lag_30', 'is_holiday']]
        y_test = self.test['Sales']

        xgb = XGBRegressor(n_estimators=100)
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        
        score = mean_absolute_percentage_error(y_test, preds)
        
        # Save the model
        joblib.dump(xgb, "models/best_model.pkl")
        
        self.best_name = "XGBoost"
        self.best_score = score
        
        return self.best_name, self.best_score