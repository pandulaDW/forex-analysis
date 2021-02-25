""" All functions to do forecasting and calculate measures """
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import statsmodels.api as sm


def df_to_single_supervised(data: pd.DataFrame, p_rate: str, s_rate: str, n_in: int = 1) -> pd.DataFrame:
    """
    Frame a DataFrame as a single step supervised learning dataset.
    :param data: Two column dataframe containing response rate and input rate
    :param n_in: Number of lag observations as input (X).
    :param s_rate: Secondary rate
    :param p_rate: Primary rate
    :return Pandas DataFrame of series framed for supervised learning.
    """
    cols = [data.loc[:, p_rate]]
    col_names = [p_rate.lower()]

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.loc[:, s_rate].shift(i))
        col_names.append(f"{s_rate.lower()}_t-{i}")

    agg = pd.concat(cols, axis=1)
    agg.columns = col_names
    agg.dropna(inplace=True)
    return agg


def df_to_multi_supervised(data: pd.DataFrame, p_rate: str, input_rates: list) -> pd.DataFrame:
    """
    Frame a DataFrame as a single step supervised learning dataset for multiple
    input variables. Only one step shifted variables wil be used to easily assess
    feature importance
    Returns:
        Pandas DataFrame of series framed for supervised learning.
        :param data: Dataframe containing response rate and input rates
        :param p_rate: Primary rate
        :param input_rates: Input rates
    """
    cols = [data.loc[:, p_rate]]
    col_names = [p_rate.lower()]

    for rate in input_rates:
        cols.append(data.loc[:, rate].shift(1))
        col_names.append(f"{rate.lower()}_t-{1}")

    agg = pd.concat(cols, axis=1)
    agg.columns = col_names
    agg.dropna(inplace=True)
    return agg


def train_test_split(data: np.ndarray, n_test: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split the dataset into train and test, by using the
    the number of test data points """
    return data[:-n_test, :], data[-n_test:, :]


def xgboost_forecast(train: List[np.ndarray], test_X: np.ndarray) -> tuple:
    """
    XGboost regression on data, will only predict one step
    Expects the first column to be the response column
    """
    train = np.asarray(train)  # convert list of rows back to a 2d np array
    train_X, train_y = train[:, 1:], train[:, 0]
    model = XGBRegressor(objective="reg:squarederror",
                         n_estimators=300, n_jobs=-1)
    model.fit(train_X, train_y)
    y_hat = model.predict(test_X)
    return y_hat, model.feature_importances_


def two_step_forecasting(data: np.ndarray) -> tuple:
    """
    Two step training to forecast 50 data points
    expects the first column to be the response column
    """
    predictions = list()
    feature_importance_list = list()

    # split dataset
    train, test = train_test_split(data, 50)

    # create a list of input rows
    history = [x for x in train]

    for round_ in range(2):
        if round_ == 0:
            testX = test[0:25, 1:]
        else:
            testX = test[25:, 1:]

        # fit model on history and make a prediction
        y_hat, feature_importance = xgboost_forecast(history, testX)

        # store forecast in list of predictions
        predictions = [*predictions, *y_hat]

        # Store feature importance for each iteration
        feature_importance_list.append(feature_importance)

        # add actual observation to history for the next loop
        if round_ == 0:
            history = [*history, *test[0:25].tolist()]

    error = mean_absolute_error(test[:, 0], predictions)
    return error, test[:, 0], predictions, feature_importance_list


def get_kde_estimations(y: np.ndarray, y_hat: list) -> Dict[str, list]:
    """ Return Kernal Density Estimation Values """
    residuals = y - y_hat
    kde = sm.nonparametric.KDEUnivariate(residuals)
    kde.fit()  # Estimate the densities
    return {"support": kde.support.tolist(), "density": kde.density.tolist()}


def get_mean_feature_importance(feature_importance_list: list, input_rates: list) -> Dict[str, float]:
    """ Return mean feature importance with columns """
    means = np.asarray(feature_importance_list).mean(axis=0)
    return {input_rates[i]: float(means[i]) for i in range(len(input_rates))}
