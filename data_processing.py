import pathlib
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from dateutil import parser
from sklearn.preprocessing import MinMaxScaler
from stat_calculation import calculate_all_stats

data_path = pathlib.Path("data")
stat_file_path = pathlib.Path("results", "stats.json")


def convert_to_date(row):
    """
    Convert date column to a proper format
    """
    if type(row["Date"]) is datetime:
        return parser.parse(str(row["Date"]), dayfirst=True)
    else:
        return parser.parse(row["Date"], dayfirst=False)


def read_and_process_interest_data() -> DataFrame:
    """
    Read the interest data and fix the date column
    """
    df_interest = pd.read_excel(data_path.joinpath(
        "Daily Feds Treasury yield .xlsx"), parse_dates=False)
    df_interest.dropna(inplace=True)
    date_values = []

    for _, row in df_interest.iterrows():
        date_values.append(convert_to_date(row))

    df_interest["Date"] = date_values
    df_interest.sort_values(by="Date", ascending=False, inplace=True)
    df_interest.columns = ["BaseDate", "T_YIELD"]
    return df_interest


def read_and_process_data() -> DataFrame:
    """
    Read the initial excel file, scale the data, write it to disk
    and returns it
    """
    df = pd.read_excel(data_path.joinpath("Spot prices_Africa.xlsx"),
                       header=1, parse_dates=["BaseDate"])
    df_interest = read_and_process_interest_data()
    df_final = pd.merge(df, df_interest, on="BaseDate")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_final.iloc[:, 1:])
    df_final.iloc[:, 1:] = scaled_data

    df_final.to_csv(data_path.joinpath("scaled_data.csv"), index=False)
    return df_final


def write_stats():
    """
    Write stats to disk
    """
    df = read_and_process_data()
    calculate_all_stats(df, stat_file_path.absolute())


if __name__ == "__main__":
    write_stats()
