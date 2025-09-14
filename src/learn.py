from mread import read, readdir
from features import calc_aggregate_features
from readydata import ready_transactions, ready_transfers, ready_clients, merge_data

import lightgbm as lgb
import joblib
from vars import ALL_CATS, ALL_TRANSFS

def learn():
    df_clients = read('clients.csv', frm="learn")
    df_transactions = readdir('transactions', frm="learn")
    df_transfers = readdir('transfers', frm="learn")

    targets = df_transactions[['client_code', 'product']].drop_duplicates().rename(columns={'product': 'target'})

    data = merge_data(calc_aggregate_features(ready_transactions(df_transactions), 'category', ALL_CATS), calc_aggregate_features(ready_transfers(df_transfers), 'type', ALL_TRANSFS))
    data = merge_data(data, ready_clients(df_clients))
    data = merge_data(data, targets)


    data.to_csv("data.csv", index=False, encoding="utf-8-sig")


    x = data.drop(columns=["target"])
    x['status'] = x['status'].astype('category')
    x['city'] = x['city'].astype('category')

    client_codes = x["client_code"]


    x = x.drop(columns=["client_code"])
    y = data["target"]

    x.to_csv("x2.csv", index=False, encoding="utf-8-sig")

    model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="multiclass",
        num_class=y.nunique(),
        min_data_in_leaf=1,
        random_state=42
    )

    model.fit(x, y, categorical_feature=['status', 'city'])

    joblib.dump(model, "model.pkl")