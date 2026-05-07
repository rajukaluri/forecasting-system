import pandas as pd
import numpy as np

class TimeSeriesProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_and_clean(self):
        # 1. Load data
        self.df = pd.read_excel(self.file_path)
        
        # 2. Convert Date to datetime object
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # 3. Aggregating by Date (This is usually where 'State' disappears)
        # We group by Date and Sum Sales to get a single timeline
        self.df = self.df.groupby('Date')['Sales'].sum().reset_index()
        
        # 4. Sort by date
        self.df = self.df.sort_values('Date')
        
        return self

    def add_features(self):
        # Create the Lag features required for your assignment
        self.df['lag_1'] = self.df['Sales'].shift(1)
        self.df['lag_7'] = self.df['Sales'].shift(7)
        self.df['lag_30'] = self.df['Sales'].shift(30)
        
        # Add a simple holiday/weekend flag
        self.df['is_holiday'] = self.df['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        
        # Drop rows with NaN created by lagging
        self.df = self.df.dropna()
        return self.df

    def split_data(self, data):
        # Simple 80/20 split for time series
        split_point = int(len(data) * 0.8)
        train = data.iloc[:split_point]
        test = data.iloc[split_point:]
        return train, test