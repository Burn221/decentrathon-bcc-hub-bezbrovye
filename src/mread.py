import pandas as pd
import glob

def read(file: str) -> pd.DataFrame:
    return pd.read_csv(f'data/{file}', dtype={"client_code": str})

def readdir(dir: str) -> pd.DataFrame:
    files: list = glob.glob(f'./data/{dir}/*.csv')
    datas: list = [read(file.replace('./data/', '')) for file in files]
    result: pd.DataFrame = pd.concat(datas, ignore_index=True)
    return result
