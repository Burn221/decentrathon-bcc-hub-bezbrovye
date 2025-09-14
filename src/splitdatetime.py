def splitdatetime(dfield):
    data_transactions['day'] = data_transactions['date'].dt.day
    data_transactions['month'] = data_transactions['date'].dt.month
    data_transactions['year'] = data_transactions['date'].dt.year
    data_transactions['time'] = data_transactions['date'].dt.time
    data_transactions['hour'] = data_transactions['date'].dt.hour
    data_transactions['minute'] = data_transactions['date'].dt.minute
    data_transactions['second'] = data_transactions['date'].dt.second