import pandas as pd
from vars import CATEGORIES
from util import classify_city

def ready_transactions(dataframe):
    dataframe['category'] = dataframe['category'].map(CATEGORIES).fillna('other')

    dataframe['date'] = pd.to_datetime(dataframe['date'], errors='coerce')
    dataframe['day'] = dataframe['date'].dt.day
    dataframe['month'] = dataframe['date'].dt.month
    dataframe['year'] = dataframe['date'].dt.year
    dataframe['time'] = dataframe['date'].dt.time
    dataframe['hour'] = dataframe['date'].dt.hour
    dataframe['minute'] = dataframe['date'].dt.minute
    dataframe['second'] = dataframe['date'].dt.second

    dataframe['day_type'] = dataframe['date'].dt.weekday.map(
        lambda x: "workweek" if x < 5 else "weekend"
    )
    dataframe['amount'] = pd.to_numeric(dataframe['amount'], errors='coerce')
    
    return dataframe

def ready_transfers(dataframe):
    dataframe['date'] = pd.to_datetime(dataframe['date'], errors='coerce')
    dataframe['amount'] = pd.to_numeric(dataframe['amount'], errors='coerce')

    return dataframe

def ready_clients(dataframe):
    dataframe = dataframe.drop(columns=["name"])
    dataframe['city'] = dataframe['city'].map(classify_city)

    return dataframe

def merge_data(datafield_1, datafield_2):
    merged = pd.merge(
        datafield_1,
        datafield_2,
        on="client_code",
        how="inner"
    )
    return merged
