import pandas as pd


def day_to_week_dict(calendar_file_path: str = 'data/raw/calendar.csv'):
    calendar = pd.read_csv(calendar_file_path, index_col='d')
    calendar = calendar['wm_yr_wk']
    map_dict = calendar.to_dict()
    return map_dict


def select_tail_days(ds: pd.DataFrame, length=28, save_rest_columns=True):
    all_days = ds.columns[ds.columns.str.startswith('d_')].tolist()
    rest_columns = [col for col in ds.columns if col not in all_days] if save_rest_columns else []
    tail_days = all_days[-length:]
    ds_tail_days = ds[rest_columns + tail_days]
    return ds_tail_days


def only_days_columns(ds: pd.DataFrame):
    return ds.columns[ds.columns.str.startswith('d_')].tolist()
