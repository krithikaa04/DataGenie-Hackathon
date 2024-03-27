import logging
import multiprocessing
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from scipy.stats import kurtosis, skew
from statsmodels.tsa.seasonal import seasonal_decompose

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
warnings.filterwarnings("ignore")

def preprocess_data(df, format, from_date, to_date):
    model_input= []

    mean = df['value'].mean()
    variance = df['value'].var()
    model_input.append(mean)
    model_input.append(variance)

    skewness =  skew(df["value"])
    model_input.append(skewness)
    kurt =  kurtosis(df["value"])
    model_input.append(kurt)

    if (
        pd.to_datetime(from_date).date() != df["date"][0].date()
        or pd.to_datetime(to_date).date() != df["date"][-1].date()
    ):
        return (
            None,
            {
                "message": "Date mentioned in the parameters and payload is not matching!"
            },
        )
    try:
        result = seasonal_decompose(
            df["value"], model="multiplicative", extrapolate_trend="freq"
        )
    except:
        result = seasonal_decompose(
            df["value"],
            model="additive",
            extrapolate_trend="freq",
            period=1,
        )
    trend_mean = result.trend.mean()
    model_input.append(trend_mean)
    seasonal_mean = result.seasonal.mean()
    model_input.append(seasonal_mean)
    if format=="weekly" or format=="hourly" or format=="monthly":
        residual_mean = result.resid.mean()
        model_input.append(residual_mean)
    return model_input, None

def calculate_mape(true_values, predicted_values):
    epsilon = 1e-10
    abs_percentage_error = np.abs(
        (true_values - predicted_values) / (true_values + epsilon)
    )
    return round(np.mean(abs_percentage_error) * 100, 4)

def optimal_threshold(forecast_score):
    threshold_mean = np.mean(forecast_score)
    threshold_std = np.std(forecast_score)
    forecast_score_z_score = (forecast_score - threshold_mean) / threshold_std
    threshold_z = 1.5
    selected_forecast_scores = forecast_score[forecast_score_z_score >= threshold_z]
    return selected_forecast_scores

def predict_values(df, period=0, date_to=None):    
    df = df.rename(columns={"date": "ds", "value": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    mape = calculate_mape(df["y"], forecast["yhat"].tolist()[: len(df)])
    return forecast["yhat"].tolist(),mape 

def prepare_data(data):
    data = data.rename(columns={'date': 'ds', 'value': 'y'})
    return data

def rolling_window(data, window_size, step=1):
    data = data.reset_index(drop=True)
    length = data.shape[0]
    total_steps = ((length - window_size) // step) + 1
    for i in range(total_steps):
        start = i * step
        end = min(start + window_size, length)
        if end - start >= window_size:
            yield window_size, data[start:end]

def batch_anomaly_detection(data, window_size, initial_batch_size=None):
    if initial_batch_size is None:
        initial_batch_size = min(len(data) * 0.1, 36)
    print("sample data:",data)
    data = data.rename(columns={"date": "ds", "value": "y"})

    num_batches = 0
    num_batch_fits = 0
    total_batch_fit_time = 0

    for (_, batch_data) in rolling_window(data, window_size, step=1):
        prophet_model = Prophet()
        start_time = time.time()
        prophet_model.fit(batch_data)
        end_time = time.time()

        num_batches += 1
        num_batch_fits += 1
        total_batch_fit_time += (end_time - start_time)
    avg_batch_fit_time = total_batch_fit_time / num_batch_fits
    return num_batch_fits, avg_batch_fit_time

#The below written code is the one to reduce the average batch fit time by parallely executing the prophet models for each batch. But due to the heavy load, I wasn't able to run on my system :(
'''
def rolling_window(data, window_size, step=1):
    data = data.reset_index(drop=True)
    length = data.shape[0]
    total_steps = ((length - window_size) // step) + 1
    for i in range(total_steps):
        start = i * step
        end = min(start + window_size, length)
        if end - start >= window_size:
            yield window_size, data[start:end]

def fit_single_batch(batch_data):
    prophet_model = Prophet()
    return prophet_model.fit(batch_data)

def fit_model_in_batches(data_lists, n_jobs=-1):
    if n_jobs == 1:
        return [fit_single_batch(batch_data) for batch_data in data_lists]

    models = []
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    processes = []

    for batch_data in data_lists:
        p = multiprocessing.Process(target=fit_and_store_model, args=(batch_data, result_dict))
        p.start()
        processes.append(p)

    for i, _ in enumerate(processes):
        result_dict.get(i, None)

    for process in processes:
        process.join()

    for model_num in result_dict.keys():
        models.append(result_dict[model_num])

    return models

def fit_and_store_model(batch_data, result_dict):
    prophet_model = fit_single_batch(batch_data)
    result_dict[len(result_dict)] = prophet_model

def batch_anomaly_detection(data, window_size, initial_batch_size=None):
    data = prepare_data(data)
    data_lists = list(rolling_window(data, window_size, step=1))

    if initial_batch_size is None:
        initial_batch_size = min(len(data) * 0.1, 36)

    model_fits = fit_model_in_batches(data_lists, n_jobs=-1)
    num_batches = len(model_fits)
    total_batch_fit_time = 0

    for prophet_model in model_fits:
        start_time = time.time()
        prophet_model.fit(data)
        end_time = time.time()
        total_batch_fit_time += (end_time - start_time)

    avg_batch_fit_time = total_batch_fit_time / num_batches

    return num_batches, avg_batch_fit_time
'''