import joblib
import os
import pandas as pd

from utils.dtype import downcast


def read_sales_dataset(path: str = 'data/raw/sales_train_evaluation.csv', use_cache=True):
    if use_cache and os.path.isfile('data/cache/sales.jbl'):
        ds = joblib.load('data/cache/sales.jbl')
        return ds

    ds = pd.read_csv(path)

    ds.loc[:, 'constant_id'] = 0

    ds = downcast(ds)
    categorical_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    ds[categorical_columns] = ds[categorical_columns].astype('category')

    if use_cache:
        joblib.dump(ds, 'data/cache/sales.jbl')

    return ds


def read_calendar_dataset(path: str = 'data/raw/calendar.csv', use_cache=True):
    if use_cache and os.path.isfile('data/cache/calendar.jbl'):
        ds = joblib.load('data/cache/calendar.jbl')
        return ds

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

    if use_cache:
        joblib.dump(ds, 'data/cache/calendar.jbl')

    return ds


def read_prices_dataset(path: str = 'data/raw/sell_prices.csv', use_cache=True):
    if use_cache and os.path.isfile('data/cache/prices.jbl'):
        ds = joblib.load('data/cache/prices.jbl')
        return ds

    ds = pd.read_csv(path)
    ds = downcast(ds)
    categorical_columns = ['store_id', 'item_id']
    ds[categorical_columns] = ds[categorical_columns].astype('category')

    if use_cache:
        joblib.dump(ds, 'data/cache/prices.jbl')

    return ds