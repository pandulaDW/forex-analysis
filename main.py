import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# pylint: disable=no-name-in-module
from pydantic import BaseModel

from data_processing import data_path, stat_file_path
from forecasting import (df_to_single_supervised, two_step_forecasting, get_kde_estimations, df_to_multi_supervised,
                         get_mean_feature_importance)
from stat_calculation import calculate_full_corr, calculate_rolling_window_median, calculate_rolling_window_corr

app = FastAPI()
data_path = Path("data")
stat_file_path = Path("results", "stats.json")

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# load the written data
scaled_df = pd.read_csv(data_path.joinpath("scaled_data.csv"))
cluster_df = pd.read_csv(data_path.joinpath("final.csv"))
with open(stat_file_path) as read_file:
    stat = json.load(read_file)


@app.get("/forex/scaled_data")
def send_scaled_data():
    data_dict = scaled_df.to_dict()
    return JSONResponse(content=data_dict)


@app.get("/forex/computed_stats")
def send_computed_stats():
    return JSONResponse(content=stat)


@app.get("/forex/corr_chart_values")
def send_chart_values(p_rate: str, s_rate: str):
    full_corr = calculate_full_corr(scaled_df, p_rate, s_rate)
    rolling_median_primary = calculate_rolling_window_median(scaled_df, p_rate)
    rolling_median_secondary = calculate_rolling_window_median(
        scaled_df, s_rate)
    rolling_window_corr = calculate_rolling_window_corr(
        scaled_df, p_rate, s_rate)
    return {"full_corr": full_corr, "rolling_median_primary": rolling_median_primary,
            "rolling_median_secondary": rolling_median_secondary, "rolling_corr": rolling_window_corr}


@app.get("/forex/single_forecasting")
def calculate_single_forecasts(p_rate: str, s_rate: str):
    """
    Calculate p_rate using 3 lagged s_rate for 20 points
    """
    input_df = df_to_single_supervised(scaled_df, p_rate, s_rate, n_in=3)
    X = input_df.values
    mae, y, y_hat, _ = two_step_forecasting(X)
    kde_result = get_kde_estimations(y, y_hat)
    return {"mae": mae, "actual": y.tolist(), "predictions": np.array(y_hat).tolist(), "kde_result": kde_result}


class MultiForecastBody(BaseModel):
    p_rate: str
    input_rates: List[str]


@app.post("/forex/multi_forecasting")
def calculate_multi_forecasts(req: MultiForecastBody):
    """
    Calculate p_rate using 1 lagged input rates for 20 points
    """
    input_df = df_to_multi_supervised(scaled_df, req.p_rate, req.input_rates)
    X = input_df.values
    mae, y, y_hat, feature_importance_list = two_step_forecasting(X)
    kde_result = get_kde_estimations(y, y_hat)
    feature_importance = get_mean_feature_importance(
        feature_importance_list, req.input_rates)
    return {"mae": mae, "actual": y.tolist(), "predictions": np.array(y_hat).tolist(), "kde_result": kde_result,
            "feature_importance": feature_importance}


@app.get("/forex/cluster_results")
def get_cluster_results():
    """
    Get full cluster list
    """
    return {row["currency"]: row["clusters"] for _, row in cluster_df.iterrows()}