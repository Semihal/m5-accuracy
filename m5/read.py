import logging

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
    categorical_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    ds[categorical_columns] = ds[categorical_columns].astype('category')
    ds = downcast(ds)

    if use_cache:
        joblib.dump(ds, 'data/cache/sales.jbl')
    return ds


def read_calendar_dataset(path: str = 'data/raw/calendar.csv', use_cache=True):
    if use_cache and os.path.isfile('data/cache/calendar.jbl'):
        ds = joblib.load('data/cache/calendar.jbl')
        return ds

    ds = pd.read_csv(path, date_parser=['date'])
    ds = ds.set_index('d')
    categorical_columns = [
        'weekday',
        'event_name_1', 'event_type_1',
        'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI'
    ]
    ds[categorical_columns] = ds[categorical_columns].astype('category')
    ds = downcast(ds)

    if use_cache:
        joblib.dump(ds, 'data/cache/calendar.jbl')
    return ds


def read_prices_dataset(path: str = 'data/raw/sell_prices.csv', use_cache=True):
    if use_cache and os.path.isfile('data/cache/prices.jbl'):
        ds = joblib.load('data/cache/prices.jbl')
        return ds

    ds = pd.read_csv(path)
    categorical_columns = ['store_id', 'item_id']
    ds[categorical_columns] = ds[categorical_columns].astype('category')
    ds = downcast(ds)

    if use_cache:
        joblib.dump(ds, 'data/cache/prices.jbl')
    return ds


def build_base_dataset(use_cache=True):
    if use_cache and os.path.isfile('data/cache/dataset.jbl'):
        logging.info('Use cache')
        ds = joblib.load('data/cache/dataset.jbl')
        return ds

    logging.info('Re-building the basic data set.')
    eval_set = read_sales_dataset()
    cal = read_calendar_dataset()
    prices_set = read_prices_dataset()

    sold = eval_set.melt(
        id_vars=['constant_id', 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        value_vars=[col for col in eval_set.columns if col.startswith('d_')],
        var_name='d',
        value_name='sold'
    )
    ds = sold.merge(cal, on='d', copy=False)
    ds = ds.merge(prices_set, on=['store_id', 'item_id', 'wm_yr_wk'], copy=False)

    ds['d'] = ds['d'].apply(lambda x: x.split('_')[1]).astype('int16')
    ds['date'] = pd.to_datetime(ds['date'], format='%Y-%m-%d')

    if use_cache:
        joblib.dump(ds, 'data/cache/dataset.jbl')

    return ds