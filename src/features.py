import pandas as pd
from vars import CATEGORIES

def calcAggregateFeatures(dataframe: pd.DataFrame, col) -> pd.DataFrame:
    result = dataframe.pivot_table(
        index='client_code',
        columns=f'{col}',
        values='amount',
        aggfunc=['sum', 'mean', 'count'],
        fill_value=0
    )
    result.columns = [f"{CATEGORIES.get(cat, cat)}_{stat}" for stat, cat in result.columns.to_flat_index()]
    return result.reset_index()