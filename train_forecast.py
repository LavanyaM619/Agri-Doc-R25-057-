import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import preprocess_data as pda
import Train_evaluate as TV
import warnings
warnings.filterwarnings('ignore')


def main():
    # Generate data if missing
    
    print("Loading and preprocessing data...")
    df = pda.load_and_preprocess_data()
    
    # Split data
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    print("\nTraining Price Forecasting Model...")
    price_model = TV.train_arima_model(train['price'], order=(2,1,2))
    price_mae, price_rmse, price_preds = TV.evaluate_model(price_model, test['price'])
    
    print("\nTraining Demand Forecasting Model...")
    demand_model = TV.train_arima_model(train['demand'], order=(1,1,1))
    demand_mae, demand_rmse, demand_preds = TV.evaluate_model(demand_model, test['demand'])
    
    # Save models
    joblib.dump(price_model, 'price_forecast_model.pkl')
    joblib.dump(demand_model, 'demand_forecast_model.pkl')
    
    # Evaluation results
    print("\nModel Evaluation Results:")
    print(f"Price Model - MAE: {price_mae:.2f}, RMSE: {price_rmse:.2f}")
    print(f"Demand Model - MAE: {demand_mae:.2f}, RMSE: {demand_rmse:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(test.index, test['price'], label='Actual Price')
    plt.plot(test.index, price_preds, label='Predicted Price')
    plt.title('Price Forecasting')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test.index, test['demand'], label='Actual Demand')
    plt.plot(test.index, demand_preds, label='Predicted Demand')
    plt.title('Demand Forecasting')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()