import os

import joblib
import pandas as pd
import lightgbm as lgb

from m5.model import train
from utils.read import read_sales_dataset, read_calendar_dataset, read_prices_dataset
from utils.dtype import fix_merge_dtypes


def build_base_dataset(use_cache=True):
    if use_cache and os.path.isfile('data/cache/dataset.jbl'):
        ds = joblib.load('data/cache/dataset.jbl')
        return ds

    eval_set = read_sales_dataset()
    cal = read_calendar_dataset()
    prices_set = read_prices_dataset()

    sold = eval_set.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'constant_id'],
        var_name='d',
        value_name='sold'
    ).dropna()
    ds = pd.merge(sold, cal, left_on='d', right_index=True, how='left')
    ds = pd.merge(ds, prices_set, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    ds['d'] = ds['d'].apply(lambda x: x.split('_')[1]).astype('int16')
    ds = fix_merge_dtypes(ds)

    if use_cache:
        joblib.dump(ds, 'data/cache/dataset.jbl')

    return ds


CATEGORICAL_FEATURES = [
    'dept_id', 'cat_id', 'store_id', 'state_id',
    'weekday', 'month', 'year',
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
    'snap_CA', 'snap_TX', 'snap_WI'
]
NUMERICAL_FEATURES = []


if __name__ == '__main__':
    XS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    TARGET = 'sold'

    ds = build_base_dataset()
    train_ds = ds[ds['d'] < 1914]
    val_ds = ds[ds['d'].between(1914, 1942)]

    train_set = lgb.Dataset(train_ds[XS], train_ds[TARGET], categorical_feature=CATEGORICAL_FEATURES)
    valid_set = lgb.Dataset(val_ds[XS], val_ds[TARGET], categorical_feature=CATEGORICAL_FEATURES)
