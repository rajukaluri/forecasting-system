 # 📈 Sales Forecasting Service:
 
 End-to-End MLOps SystemThis project is a production-ready forecasting system designed to predict 8-week sales trends across multiple states. It features an automated Model Tournament that selects the best-performing algorithm and deploys it as a FastAPI web service.🏗️ Project ArchitectureThe system is designed with a modular backend architecture to separate data processing, model training, and inference.

Plaintextforecasting_system/
│
├── data/                       
│   └── sales_data.xlsx         
│
├── models/                     
│   └── best_model.pkl          
│
├── src/                        
│   ├── data_processor.py       
│   ├── trainer.py              
│   ├── model_utils.py          
│   └── main.py                 
│
├── README.md                    
├── requirements.txt            
└── run_pipeline.py             

 # 🛠️ Feature Engineering
 
 (The Logic)The system satisfies the project requirements by engineering temporal features that capture patterns without data leakage:Recursive Lags: 
 
 Creates $t-1$, $t-7$, and $t-30$ features to capture daily, weekly, and monthly seasonality.Rolling Statistics: Calculates 7-day and 30-day rolling means and standard deviations to capture volatility.Temporal Flags: Extracts Day of Week, Month, and is_holiday from timestamps.

 # 🏆 Model Tournament
 
 The system automatically trains and compares four distinct models using MAPE (Mean Absolute Percentage Error) as the selection metric:SARIMA: Statistical modeling for seasonal trends.Facebook Prophet: Additive modeling for holiday and trend shifts.XGBoost: Gradient boosting using engineered lag features.LSTM: Deep learning for long-term sequential dependencies.

 # 🚀 Execution Guide

1. Environment SetupBash# Install dependencies
pip install -r requirements.txt

2. Run the Training PipelineThis command triggers the data processing, runs the tournament, and saves the winner to models/.Bash# Ensure PYTHONPATH is set to the current directory
$env:PYTHONPATH = "."; python run_pipeline.py

3. Deploy the APILaunch the FastAPI server:Bashuvicorn src.main:app --reload
🌐 API InteractionOnce the API is live, visit http://127.0.0.1:8000/docs to access the interactive Swagger UI.Example Input:JSON{
  "lag_1": 5000.0,
  "lag_7": 4850.0,
  "lag_30": 4600.0,
  "is_holiday": 0
}
Example Output:JSON{
  "status": "success",
  "predicted_sales": 5124.85,
  "model_used": "XGBoost"
}
