import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    """
    Calculates performance metrics for model comparison.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    # You can add RMSE or MAE here if needed
    return mape

def create_lstm_sequences(data, window_size=1):
    """
    Reshapes flat data into a 3D array for LSTM: (samples, time_steps, features).
    Example: [10, 20, 30] with window 1 becomes [[10]], [[20]], [[30]].
    """
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape to (samples, time_steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def get_holiday_list():
    """
    Utility to return major holiday dates if you choose 
    to use the 'holidays' library for a more robust flag.
    """
    # Simple static list for assignment purposes
    return [
        '2023-12-25', '2024-12-25', # Christmas
        '2023-11-23', '2024-11-28', # Thanksgiving
        '2024-01-01'                # New Year
    ]