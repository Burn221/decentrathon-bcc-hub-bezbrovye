from vars import CATEGORIES
from mread import read, readdir
import pandas as pd
from features import calcAggregateFeatures

def main():
    data_clients = read('clients.csv')
    data_transfers = readdir('transfers')
    data_transactions = readdir('transactions')

    data_transactions['date'] = pd.to_datetime(data_transactions['date'], errors='coerce')
    data_transactions['day'] = data_transactions['date'].dt.day
    data_transactions['month'] = data_transactions['date'].dt.month
    data_transactions['year'] = data_transactions['date'].dt.year
    data_transactions['time'] = data_transactions['date'].dt.time
    data_transactions['hour'] = data_transactions['date'].dt.hour
    data_transactions['minute'] = data_transactions['date'].dt.minute
    data_transactions['second'] = data_transactions['date'].dt.second
    
    data_transfers['date'] = pd.to_datetime(data_transfers['date'], errors='coerce')

    calcAggregateFeatures(data_transactions).to_csv('output.csv', encoding='utf-8-sig')
    calcAggregateFeatures(data_transfers).to_csv('output.csv', encoding='utf-8-sig')
    
if __name__ == "__main__":
    main()



