import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_models():
    try:
        price_model = joblib.load('price_forecast_model.pkl')
        demand_model = joblib.load('demand_forecast_model.pkl')
        return price_model, demand_model
    except FileNotFoundError:
        print("Models not found. Please train models first using train_forecast.py")
        return None, None


def forecast_next_7_days(price_model, demand_model):

    price_forecast = price_model.forecast(steps=7)
    demand_forecast = demand_model.forecast(steps=7)

    # Create date
    today = datetime.now().date()
    dates = [today + timedelta(days=i) for i in range(7)]

    return pd.DataFrame({
        'date': dates,
        'price_forecast': price_forecast,
        'demand_forecast': demand_forecast
    })


def main():
    price_model, demand_model = load_models()
    if price_model is None or demand_model is None:
        return

    # Forecast next 7 days
    forecast_df = forecast_next_7_days(price_model, demand_model)

    print("\nNext 7 Days Forecast :")
    print(forecast_df.to_string(index=False))

    # Calculate percentage changes
    forecast_df['price_change%'] = forecast_df['price_forecast'].pct_change() * 100
    forecast_df['demand_change%'] = forecast_df['demand_forecast'].pct_change() * 100

    print("\nForecast with Percentage Changes:")
    print(forecast_df[['date', 'price_forecast', 'price_change%',
                       'demand_forecast', 'demand_change%']].to_string(index=False))


if __name__ == "__main__":
    main()