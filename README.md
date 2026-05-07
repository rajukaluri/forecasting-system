📈 Sales Forecasting Service: End-to-End PipelineThis repository hosts a production-grade machine learning system designed to forecast state-level sales data. It automates the transition from raw data ingestion to a live REST API, utilizing a "Model Tournament" to ensure the highest predictive accuracy.📂 Project StructureA clean, modular structure is maintained to ensure the system is easy to maintain and deploy.Plaintextforecasting_system/
│
├── data/                       # Storage for input datasets
│   └── sales_data.xlsx         # Raw historical sales data
│
├── models/                     # Production-ready model storage
│   └── best_model.pkl          # The winning model from the tournament
│
├── src/                        # Core Source Code
│   ├── __init__.py             # Makes 'src' a Python package
│   ├── data_processor.py       # Cleaning, Aggregation, & Feature Engineering
│   ├── trainer.py              # Tournament logic (SARIMA, Prophet, XGBoost, LSTM)
│   ├── model_utils.py          # Custom metrics (MAPE) and evaluation helpers
│   └── main.py                 # FastAPI application for model serving
│
├── tests/                      # Unit tests for data and model logic
├── .gitignore                  # Excludes caches, venv, and local temp files
├── README.md                   # Technical documentation and setup guide
├── requirements.txt            # List of dependencies for environment setup
├── run_pipeline.py             # Main entry point to trigger training
🛠️ Components Description1. Data Processor (data_processor.py)Responsible for transforming raw data into a model-ready format.Cleaning: Handles missing values and date alignment.Feature Engineering: Generates Lags ($t-1, t-7, t-30$), Rolling Statistics (7-day mean), and Temporal Features (Day of week, Holidays).2. Model Tournament (trainer.py)Automatically evaluates four distinct algorithms to find the best fit for the current data:SARIMA: For linear seasonal trends.Prophet: For robust handling of outliers and holidays.XGBoost: For non-linear relationships using engineered lags.LSTM: Deep learning for complex sequential dependencies.3. Inference API (main.py)Built with FastAPI, this script loads the serialized best_model.pkl and exposes a /predict endpoint. It validates incoming JSON requests using Pydantic to ensure high reliability.🚀 Quick Start1. Setup EnvironmentBashpython -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
2. Run TrainingBashpython run_pipeline.py
3. Launch APIBashuvicorn src.main:app --reload
