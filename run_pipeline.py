import os
import pandas as pd
from src.data_processor import TimeSeriesProcessor
from src.trainer import ForecastTrainer

def start_system():
    # 1. Ensure the models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 2. Setup Data Path - Updating to your file name
    data_path = 'data/sales_data.xlsx' 
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Could not find {data_path}")
        return

    # --- DEBUG STEP: Check columns before processing ---
    df_check = pd.read_excel(data_path)
    print(f"🔍 Checking Excel structure...")
    print(f"Found Columns: {df_check.columns.tolist()}")
    
    # Check if 'State' exists (case-sensitive)
    if 'State' not in df_check.columns:
        print("❌ ERROR: Column 'State' not found.")
        print("Hint: Check if it is lowercase 'state' or has spaces like ' State'.")
        print("Please fix the Excel header and run again.")
        return
    # ------------------------------------------------

    # 3. Process Data
    print("\n--- Phase 1: Data Processing & Feature Engineering ---")
    try:
        processor = TimeSeriesProcessor(data_path)
        
        # Load and clean
        full_data = processor.load_and_clean().add_features()
        train, test = processor.split_data(full_data)
        
        print(f"✅ Data prepared. Training records: {len(train)}")

        # 4. Model Tournament
        print("\n--- Phase 2: Model Training Tournament ---")
        print("Testing: SARIMA, Prophet, XGBoost, and LSTM...")
        
        trainer = ForecastTrainer(train, test)
        best_name, best_score = trainer.run_all()
        
        print("\n" + "="*40)
        print(f"🏆 TOURNAMENT WINNER: {best_name}")
        print(f"📉 BEST MAPE SCORE: {best_score:.4f}")
        print("="*40)
        print("\n✅ System ready. Next step: Start the API using:")
        print("uvicorn src.main:app --reload")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during processing: {e}")

if __name__ == "__main__":
    start_system()