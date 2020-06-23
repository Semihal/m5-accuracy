from typing import Dict
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

from m5.read import build_base_dataset
from m5.funcs import only_days_columns
from m5.constants import *


logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


class WRMSSE:

    def __init__(self, train, validate, target_col='sold', levels=LEVELS):
        self.target_col = target_col
        logging.info('Calculate profit')
        self.daily_profit = self._daily_profit(train)
        logging.info('Transform train to standard view')
        self.train = self._to_standard_view(train)
        logging.info('Transform validation to standard view')
        self.validate = self._to_standard_view(validate)
        self.validate_days = only_days_columns(self.validate)
        # self.prices = prices

        logging.info('Calculate levels specification')
        self.levels = levels
        self._levels_spec = self.levels_specifications()

    def _daily_profit(self, train) -> pd.DataFrame:
        # get last 28 days
        train = train[ID_COLUMNS + ['d', 'wm_yr_wk', 'sell_price'] + [self.target_col]]
        max_day = train['d'].max()
        ds_last_days = train[train['d'] > max_day - 28]
        # calculate revenue
        ds_day_revenue = self._revenue(ds_last_days, train)
        return ds_day_revenue

    def _revenue(self, ds_with_prices: pd.DataFrame, train: pd.DataFrame):
        # add revenue feature
        ds_with_prices.loc[:, 'revenue'] = ds_with_prices[self.target_col] * ds_with_prices['sell_price']
        ds_with_prices = ds_with_prices.set_index(['item_id', 'store_id', 'd'])['revenue']
        ds_day_revenue = ds_with_prices.unstack(level=2)

        train_index = train[ID_COLUMNS].drop_duplicates()
        ds_item_revenue = train_index.merge(ds_day_revenue, left_on=['item_id', 'store_id'], right_index=True)
        rename_dict = {day_num: f'd_{day_num}' for day_num in ds_item_revenue.columns if isinstance(day_num, int)}
        ds_item_revenue = ds_item_revenue.rename(columns=rename_dict)
        return ds_item_revenue

    def _to_standard_view(self, ds: pd.DataFrame):
        indexes = ds[ID_COLUMNS].drop_duplicates()
        pivot = pd.pivot(ds, index='id', columns='d', values=self.target_col).reset_index()
        view = pivot.merge(indexes, on='id')
        rename_dict = {day_num: f'd_{day_num}' for day_num in view.columns if isinstance(day_num, int)}
        view = view.rename(columns=rename_dict)
        return view

    def levels_specifications(self) -> Dict:
        # get columns name
        train_days = only_days_columns(self.train)
        daily_profit_days = only_days_columns(self.daily_profit)
        # make dummy specifications dict
        specifications = {
            i: {
                'denominator': [],
                'weights': [],
                'validate_targets': []
            }
            for i, _ in enumerate(LEVELS, start=1)
        }
        # calculate specifications
        for level_id, level in tqdm(enumerate(self.levels, start=1), total=len(self.levels)):
            level_series = self.train.groupby(level)[train_days].sum()
            for _, row in level_series.iterrows():
                # start of the active period (p. 6)
                active_sale_start = np.argmax(row.values != 0)
                series: pd.Series = row[active_sale_start:]
                # naive shift-predict
                difference = (series - series.shift(1)) ** 2
                difference_mean = difference.mean()
                # save denominator value
                specifications[level_id]['denominator'].append(difference_mean)

            # calculate weights
            level_profit = self.daily_profit.groupby(level)[daily_profit_days].sum()
            level_weights = level_profit.sum(axis=1)
            # save normalize weights
            level_normalize_weights = level_weights / level_weights.sum()
            specifications[level_id]['weights'] = level_normalize_weights

            # save validate targets
            validate_targets = self.validate.groupby(level)[self.validate_days].sum()
            specifications[level_id]['validate_targets'] = validate_targets

        return specifications

    def score(self, validate_pred) -> float:
        scores = []
        for level_id, level in enumerate(self.levels, start=1):
            weights = self._levels_spec[level_id]['weights']
            pred_level_series = validate_pred.groupby(level)[self.validate_days].sum()
            level_rmsse = self.rmsse(pred_level_series, level_id=level_id)
            # weighted mean save
            weighted_score = (weights * level_rmsse).sum()
            scores.append(weighted_score)
        mean_score = float(np.mean(scores))
        return mean_score

    def rmsse(self, y_pred: pd.DataFrame, level_id: int) -> float:
        validate_targets = self._levels_spec[level_id]['validate_targets']
        denominator = self._levels_spec[level_id]['denominator']
        numerator = ((validate_targets - y_pred) ** 2).mean(axis=1)
        score = (numerator / denominator).map(np.sqrt)
        return score


if __name__ == '__main__':
    ds = build_base_dataset()
    train_dataset = ds[ds['d'] <= 1913]
    val_dataset = ds[ds['d'] > 1913]
    metric = WRMSSE(train_dataset, val_dataset, levels=LEVELS[:2])
