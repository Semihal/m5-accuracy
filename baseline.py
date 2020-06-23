import os
import logging

import joblib
import pandas as pd
from tqdm import tqdm

from m5.constants import LEVELS, ID_COLUMNS
from m5.read import build_base_dataset
from utils.dtype import downcast


CACHE_DIR = os.sep.join(['data', 'cache'])
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def window_stats(ds, features, stat_funcs, levels=LEVELS, shift=28, use_cache=True):
    cache_filename: str = f"{'-'.join(features)}_{'-'.join(stat_funcs)}_s{shift}_features.jbl"
    cache_path = os.sep.join([CACHE_DIR, cache_filename])
    if use_cache and os.path.isfile(cache_path):
        return joblib.load(cache_path)

    window_stats_per_window = []
    for level in tqdm(levels):
        level = level if isinstance(level, list) else [level]
        # группируем по дням
        levels_day = level + ['d']
        day_group = ds.groupby(levels_day, as_index=False)
        # суммируем значение фичей по дням
        sum_by_group_day = day_group[features].sum().sort_values(by='d')
        level_grouped = sum_by_group_day.groupby(level)
        # список новых фичей для этой группы
        groups_mean_features = []

        # окно для подсчета среднего
        for window in [3, 7, 28]:
            # считаем агрегаты
            new_features = level_grouped[features].apply(lambda x: x.rolling(window).agg(stat_funcs).shift(28))
            columns_name = [
                f'{feature_name}_{"-".join(level)}_{aggregate}_r{window}s28'
                for (feature_name, aggregate) in new_features.columns
            ]
            # переименовываем колонки
            new_features.columns = columns_name
            # понижение типа данных (для экономия памяти)
            new_features = downcast(new_features)
            # сохраняем для последующего обзего мерджа
            sum_by_group_day = pd.concat([sum_by_group_day, new_features], axis=1)
            groups_mean_features.extend(columns_name)
        # мерджим с общим набором данных
        ds = ds.merge(sum_by_group_day[levels_day + groups_mean_features], how='left', on=levels_day)
        # пополняем общий список фичей
        window_stats_per_window.extend(groups_mean_features)

    ds = ds[ID_COLUMNS + window_stats_per_window]
    if use_cache:
        joblib.dump(ds, cache_path)

    return ds


if __name__ == '__main__':
    ds = build_base_dataset().head(1000000)
    ds, stats_features = window_stats(
        ds,
        features=['sold', 'sell_price'],
        stat_funcs=['mean', 'std', 'min', 'max'],
        levels=[
            ['constant_id']
        ],
        use_cache=False
    )
