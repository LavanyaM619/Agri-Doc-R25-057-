import pandas as pd

def load_and_preprocess_data():

    df = pd.read_csv('market_data.csv')

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Handle missing
    df = df.interpolate()

    # Feature engineering
    df['price_change'] = df['price'].pct_change()
    df['demand_change'] = df['demand'].pct_change()
    df = df.dropna()

    return df