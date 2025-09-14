from mread import read, readdir
from features import calc_aggregate_features
from readydata import ready_transactions, ready_transfers, ready_clients, merge_data

import pandas as pd
import joblib
from vars import ALL_CATS, ALL_TRANSFS

def predict():
    df_clients = read('clients.csv', frm="data")
    df_transactions = readdir('transactions', frm="data")
    df_transfers = readdir('transfers', frm="data")
    model = joblib.load("model.pkl")

    data = merge_data(calc_aggregate_features(ready_transactions(df_transactions), 'category', ALL_CATS), calc_aggregate_features(ready_transfers(df_transfers), 'type', ALL_TRANSFS))
    data = merge_data(data, ready_clients(df_clients))


    data.to_csv("data.csv", index=False, encoding="utf-8-sig")

    x = data.drop(columns=["target"])
    x['status'] = x['status'].astype('category')
    x['city'] = x['city'].astype('category')
    client_codes = x["client_code"]

    x = x.drop(columns=["client_code"])

    
    
    y_pred = model.predict(x)

    preds = pd.DataFrame({
        "client_code": client_codes,
        "pred": y_pred
    })
    preds.to_csv("preds.csv", index=False, encoding="utf-8-sig")