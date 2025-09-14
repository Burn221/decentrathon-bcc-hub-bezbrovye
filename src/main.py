from vars import CATEGORIES
from mread import read, readdir
import pandas as pd
from features import calc_aggregate_features
from readydata import ready_transactions, ready_transfers, ready_clients, merge_data

def main():
    df_clients = read('clients.csv')
    df_transactions = readdir('transactions')
    df_transfers = readdir('transfers')

    
    data = merge_data(calc_aggregate_features(ready_transactions(df_transactions), 'category'), calc_aggregate_features(ready_transfers(df_transfers), 'type'))
    data = merge_data(data, ready_clients(df_clients))

    

    data.to_csv('out.csv', encoding='utf-8-sig')
    
    # calc_aggregate_features(df_transactions, 'category').to_csv('output_transactions.csv', encoding='utf-8-sig')
    # calc_aggregate_features(df_transfers, 'type').to_csv('output_transfers.csv', encoding='utf-8-sig')
    
if __name__ == "__main__":
    main()



