import json
import numpy as np
# from dtw import dtw
from pandas import DataFrame


def calculate_correlation(df: DataFrame, primary_rate: str, secondary_rate: str) -> float:
    """
    Calculate pearson correlations
    """
    correlation = df[[primary_rate, secondary_rate]].corr().iloc[0, 1]
    return correlation


# def calculate_dtw_distance(df: DataFrame, primary_rate: str, secondary_rate: str) -> float:
#     """
#     Calculate dtw distance
#     """
#     alignment = dtw(df[primary_rate], df[secondary_rate])
#     return alignment.distance


def calculate_valid_corr(df: DataFrame, primary_rate: str, secondary_rate: str) -> float:
    """
    Calculate valid cross correlation value
    """
    valid_corr = np.correlate(df[primary_rate], df[secondary_rate])[0]
    return valid_corr


def calculate_full_corr(df: DataFrame, primary_rate: str, secondary_rate: str):
    """
    Calculate full cross correlation values
    """
    full_corr = np.correlate(df[primary_rate], df[secondary_rate], mode="full")
    return full_corr.tolist()


def calculate_rolling_window_median(df: DataFrame, rate: str) -> list:
    """
    Calculate median value at each window
    """
    median_window_size = 120
    median_r = df[rate].rolling(window=median_window_size, center=True).median().dropna()
    return median_r.tolist()


def calculate_rolling_window_corr(df: DataFrame, primary_rate: str, secondary_rate: str) -> list:
    """
    Calculate correlation value at each window
    """
    corr_window_size = 30
    rolling_r = df[primary_rate].rolling(
        window=corr_window_size, center=True).corr(df[secondary_rate]).dropna().tolist()
    rolling_r = [val for val in rolling_r if val != float("inf")]
    return [round(val, 3) for val in rolling_r]


def calculate_all_stats(df: DataFrame, stat_file_path: str):
    """
    Create and populate the stat dictionary
    """
    cols = df.columns[1:]
    stat_dict = {}
    for col in cols:
        stat_dict[col] = {rate: {} for rate in cols if rate != col}
    for p_rate in stat_dict:
        for s_rate in stat_dict[p_rate]:
            stat_dict[p_rate][s_rate]["correlation"] = calculate_correlation(df, p_rate, s_rate)
            # stat_dict[p_rate][s_rate]["dtw"] = calculate_dtw_distance(df, p_rate, s_rate)
            stat_dict[p_rate][s_rate]["cross_corr"] = calculate_valid_corr(df, p_rate, s_rate)
        print(f"Finished processing {p_rate}")
    with open(stat_file_path, "w") as write_file:
        json.dump(stat_dict, write_file)
