import pandas as pd
import numpy as np
from tqdm import tqdm


def downcast(ds: pd.DataFrame):
    dtypes_dict = ds.dtypes.to_dict()
    for col_name, col_type in tqdm(dtypes_dict.items()):
        values = ds[col_name]
        try:
            if np.issubdtype(col_type, np.integer):
                int_dtypes = [np.uint8, np.int8, np.int16, np.int32, np.int64]
                ds.loc[:, col_name] = convert_type(values, int_dtypes, np.iinfo)
            elif np.issubdtype(col_type, np.float):
                float_dtypes = [np.float16, np.float32, np.float64]
                ds.loc[:, col_name] = convert_type(values, float_dtypes, np.finfo)
        except TypeError:
            continue
    return ds


def convert_type(values, types, resolve_func):
    for dtype in types:
        if _is_type(values, dtype, resolve_func):
            # если тип установлен уже правильно
            if np.issubdtype(values.dtype, dtype):
                return values
            return values.astype(dtype)
    raise ValueError('Failed to automatically determine the best data type.')


def _is_type(values, dtype, resolve_func):
    dtype_info = resolve_func(dtype)
    return values.min() >= dtype_info.min and values.max() <= dtype_info.max


def fix_merge_dtypes(ds: pd.DataFrame):
    # to category
    categorical_columns = ['item_id', 'store_id']
    ds[categorical_columns] = ds[categorical_columns].astype('category')
    # to date
    ds['date'] = pd.to_datetime(ds['date'], format='%Y-%m-%d')
    # result
    return ds
