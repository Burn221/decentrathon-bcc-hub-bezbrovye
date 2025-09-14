import pandas as pd

def calc_aggregate_features(dataframe: pd.DataFrame, col, all_values: list) -> pd.DataFrame:
    result = dataframe.pivot_table(
        index='client_code',
        columns=f'{col}',
        values='amount',
        aggfunc=['sum', 'mean', 'count'],
        fill_value=0
    )

    result.columns = [f"{c}_{f}" for f, c in result.columns.to_flat_index()]
    result = result.reset_index()

    for value in all_values:
        for func in ['sum', 'mean', 'count']:
            colname = f"{value}_{func}"
            if colname not in result.columns:
                result[colname] = 0

    return result