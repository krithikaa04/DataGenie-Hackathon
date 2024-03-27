import logging
import pickle
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from inputData import inputData
from scipy.stats import kurtosis, skew
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from utils import batch_anomaly_detection, predict_values, preprocess_data

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
warnings.filterwarnings("ignore")

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
async def predict(
        format: str = Query(..., description="Format"),
        from_date : str = Query(..., description="Start date (YYYY-MM-DD)"),
        to_date: str = Query(..., description="End date (YYYY-MM-DD)"),
        period: int = Query(..., description="Period"),
        request_data: List[inputData] = Body(..., description="Data points for prediction"),):
        try:
            if period and period<0:
                return {"message": "period number shoudl always be greater than 0"}
            data=[]
            if request_data:
                data = [
                    {"point_timestamp": val.point_timestamp, "point_value": val.point_value}
                    for val in request_data
                ]
                data = sorted(data, key=lambda x: x["point_timestamp"])
                sample_df = pd.DataFrame(data)
                sample_df["point_timestamp"] = pd.to_datetime(sample_df["point_timestamp"])
                sample_df.columns = ["date", "value"]
                sample_df.set_index(sample_df["date"], inplace=True)
                sample_df.sort_index(inplace=True)
                sample_df.rename_axis("index", inplace=True)
                print(sample_df)
                model_input , error_msg = preprocess_data(sample_df, format, from_date, to_date)
                if format == "daily":
                    forecast_dates = pd.date_range(
                        start=pd.to_datetime(to_date) + pd.DateOffset(1),
                        periods=period,
                        freq="D",
                    )
                    with open("C:/Users/Krithika/Desktop/DataGenie-Hackathon/DG-backend/models/dailymodel.pkl", "rb") as model_file:
                        model = pickle.load(model_file)
                        print(model)
                elif format == "monthly":
                    forecast_dates = pd.date_range(
                        start=pd.to_datetime(to_date),
                        periods=period + 1,
                        freq="MS",
                    )[1:]
                    with open("C:/Users/Krithika/Desktop/DataGenie-Hackathon/DG-backend/models/monthlymodel.pkl", "rb") as model_file:
                        model = pickle.load(model_file)

                elif format == "hourly":
                    forecast_dates = pd.date_range(
                        start=sample_df["date"][-1],
                        periods=period + 1,
                        closed="left",
                        freq="H",
                    )[1:]
                    with open("C:/Users/Krithika/Desktop/DataGenie-Hackathon/DG-backend/models/hourlymodel.pkl", "rb") as model_file:
                        model = pickle.load(model_file)

                elif format == "weekly":
                    forecast_dates = pd.date_range(
                        start=pd.to_datetime(to_date) + pd.DateOffset(1),
                        periods=period,
                        freq="W",
                    )
                    with open("C:/Users/Krithika/Desktop/DataGenie-Hackathon/DG-backend/models/weeklymodel.pkl", "rb") as model_file:
                        model = pickle.load(model_file)
                forecast_score = model.predict([model_input])[0]
                y_pred, mape = predict_values(sample_df, period, to_date)
                error = np.abs(sample_df['value'] - y_pred)
                mean_error = np.mean(error)
                std_error = np.std(error)
                z_score = (error - mean_error)/std_error
                #I am keeping it as 1.5 deviations
                threshold = 1.5
                anomaly_bool = np.array(z_score>threshold)
                is_anomaly = []
                for val in anomaly_bool:
                    if val:
                        is_anomaly.append("yes")
                    else:
                        is_anomaly.append("no")
                if format=="daily" or format=="hourly":
                    window_size=7
                else:
                    window_size=4
                num_batch_fits, avg_batch_fit_time = batch_anomaly_detection(sample_df,window_size)
                dates, true_y = sample_df["date"].tolist(), sample_df["value"].tolist()
                dates.extend(forecast_dates)
                res = []
                for i in range(len(y_pred)):
                    if true_y:
                        res.append(
                            {
                                "point_timestamp": dates.pop(0),
                                "point_value": true_y.pop(0),
                                "yhat": float(round(float(y_pred[i]), 4)),
                                "isanomaly": str(is_anomaly.pop(0))
                            }
                        )
                    else:
                        res.append(
                            {
                                "point_timestamp": dates.pop(0),
                                "yhat": float(round(float(y_pred[i]), 4)),
                                "isanomaly": str(is_anomaly.pop(0))
                            }
                        )
                print(res)
                prediction_info = {
                    "forecastScore": float(forecast_score),
                    "number_of_batch_fits":(num_batch_fits),
                    "mape": float(mape),
                    "avg_time_taken_per_fit_in seconds": float(avg_batch_fit_time),
                    "result": res,
                }
                return prediction_info
            
        except Exception as e:
            print("Exception:", e)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8105)