import pandas as pd
import glob

def read(file: str, frm: str = 'data') -> pd.DataFrame:
    return pd.read_csv(f'{frm}/{file}', dtype={"client_code": str})

def readdir(dir: str, frm: str = 'data') -> pd.DataFrame:
    files: list = glob.glob(f'./{frm}/{dir}/*.csv')
    datas: list = [read(file.replace(f'./{frm}/', '')) for file in files]
    result: pd.DataFrame = pd.concat(datas, ignore_index=True)
    return result
