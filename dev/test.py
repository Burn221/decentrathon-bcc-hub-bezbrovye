import chardet

with open("./data/transactions/client_1_transactions_3m.csv", "rb") as f:
    raw = f.read(50000)  # читаем кусок
    print(chardet.detect(raw))