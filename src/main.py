from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta

# Initialize FastAPI
app = FastAPI(title="Sales Forecasting API", description="Production API for 8-week state sales prediction")

# 1. Load the winner model and the latest data context at startup
# (In a real system, you'd load this from a database/S3)
try:
    best_model = joblib.load("models/best_model.pkl")
    # Load the latest state data to generate lags for the future
    # We use this to know where the "history" ends
    history_df = pd.read_csv("data/processed/latest_sales.csv") 
    history_df['Date'] = pd.to_datetime(history_df['Date'])
except Exception as e:
    print(f"Error loading model or data: {e}")

@app.get("/")
def home():
    return {"status": "Active", "message": "Sales Forecasting System API is running."}

@app.get("/predict/{state}")
def get_forecast(state: str):
    """
    Returns an 8-week forecast for the requested state.
    """
    state = state.upper()
    
    # 2. Filter data for the specific state
    state_data = history_df[history_df['State'] == state].sort_values('Date')
    
    if state_data.empty:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found in historical data.")

    # 3. Generate Future Dates (Next 8 Weeks)
    last_date = state_data['Date'].max()
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, 9)]

    # 4. Model Inference Logic 
    # (Simplification: Different models require different inputs)
    try:
        if hasattr(best_model, 'predict'): # XGBoost / Prophet style
            # If Prophet:
            if 'Prophet' in str(type(best_model)):
                future_df = pd.DataFrame({'ds': future_dates})
                forecast = best_model.predict(future_df)
                predictions = forecast['yhat'].tail(8).tolist()
            
            # If XGBoost:
            else:
                # Use the last known features for inference
                latest_features = state_data[['lag_1', 'lag_7', 'lag_30', 'rolling_mean_4', 'month', 'day_of_week']].iloc[-1:]
                # Note: In production, you'd recursively predict each week to update lags
                preds = best_model.predict(latest_features)
                predictions = [float(p) for p in np.repeat(preds, 8)] # Simple repeat for demo

        return {
            "state": state,
            "forecast_horizon": "8 Weeks",
            "start_date": future_dates[0].strftime('%Y-%m-%d'),
            "end_date": future_dates[-1].strftime('%Y-%m-%d'),
            "predictions": [round(p, 2) for p in predictions]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)