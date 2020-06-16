import pandas as pd

from utils.dtype import downcast


def read_sales_dataset(path: str = 'data/raw/sales_train_evaluation.csv'):
    ds = pd.read_csv(path, index_col='id')

    ds.loc[:, 'constant_id'] = 0
    # Add zero sales for the remaining days 1942-1969
    for d in range(1942, 1970):
        col = 'd_' + str(d)
        ds[col] = 0

    ds = downcast(ds)
    categorical_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    ds[categorical_columns] = ds[categorical_columns].astype('category')
    return ds


def read_calendar_dataset(path: str = 'data/raw/calendar.csv'):
    ds = pd.read_csv(path, date_parser=['date'])
    ds = ds.set_index('d')
    ds = downcast(ds)
    categorical_columns = [
        'wday', 'weekday',
        'event_name_1', 'event_type_1',
        'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI'
    ]
    ds[categorical_columns] = ds[categorical_columns].astype('category')
    return ds


def read_prices_dataset(path: str = 'data/raw/sell_prices.csv'):
    ds = pd.read_csv(path)
    ds = downcast(ds)
    categorical_columns = ['store_id', 'item_id']
    ds[categorical_columns] = ds[categorical_columns].astype('category')
    return ds