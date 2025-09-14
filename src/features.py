import pandas as pd

def calc_aggregate_features(dataframe: pd.DataFrame, col) -> pd.DataFrame:
    result = dataframe.pivot_table(
        index='client_code',
        columns=f'{col}',
        values='amount',
        aggfunc=['sum', 'mean', 'count'],
        fill_value=0
    )
    result.columns = [f"{col}_{func}" for func, col in result.columns.to_flat_index()]
    return result.reset_index()