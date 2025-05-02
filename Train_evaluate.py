import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import preprocess_data as pd
import warnings
warnings.filterwarnings('ignore')

# Train ARIMA
def train_arima_model(series, order=(1,1,1)):

    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

# Evaluate model performance
def evaluate_model(model, test_data):
    predictions = model.forecast(steps=len(test_data))
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    return mae, rmse, predictions
