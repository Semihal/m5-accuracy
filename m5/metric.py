from typing import Dict

import pandas as pd
import numpy as np

from utils.funcs import day_to_week_dict, select_tail_days, only_days_columns
from m5.constants import *


def read_dataset(path: str = 'data/raw/sales_train_evaluation.csv'):
    ds = pd.read_csv(path)
    ds.loc[:, 'constant_id'] = 0
    return ds


class WRMSSE:

    def __init__(self, train, validate, prices, levels=LEVELS):
        self.train = train
        self.validate = validate
        self.validate_days = only_days_columns(self.validate)
        self.prices = prices
        self.daily_profit = self._daily_profit()

        self.levels = levels
        self._levels_spec = self.levels_specifications()

    def _daily_profit(self) -> pd.DataFrame:
        ds_used_days = select_tail_days(self.train, length=28)
        ds_with_prices = self._merge_prices(ds_used_days)
        ds_day_revenue = self._revenue(ds_with_prices)
        return ds_day_revenue

    def _merge_prices(self, ds: pd.DataFrame) -> pd.DataFrame:
        # purchase by day
        ds = ds.set_index(['item_id', 'store_id'])
        ds_only_days = ds.loc[:, ds.columns.str.startswith('d_')]
        ds_days_stack = ds_only_days.stack().reset_index()
        ds_day_purchases = ds_days_stack.rename(columns={'level_2': 'day', 0: 'purchases'})
        # add wm_yr_wk
        ds_day_purchases.loc[:, 'wm_yr_wk'] = ds_day_purchases['day'].map(day_to_week_dict())
        # merge prices
        ds_purchases_with_prices = ds_day_purchases.merge(
            self.prices, how='left',
            on=['item_id', 'store_id', 'wm_yr_wk']
        )
        return ds_purchases_with_prices

    def _revenue(self, ds_with_prices: pd.DataFrame):
        # add revenue feature
        ds_with_prices.loc[:, 'revenue'] = ds_with_prices['purchases'] * ds_with_prices['sell_price']
        #
        ds_with_prices = ds_with_prices.set_index(['item_id', 'store_id', 'day'])
        ds_day_revenue = ds_with_prices.unstack(level=2)['revenue']
        ds_day_revenue = ds_day_revenue.loc[zip(self.train.item_id, self.train.store_id), :].reset_index(drop=True)
        ds_item_revenue = pd.concat([self.train[ID_COLUMNS], ds_day_revenue], axis=1, sort=False)
        return ds_item_revenue

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
        for level_id, level in enumerate(self.levels, start=1):
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
    cal = pd.read_csv('../data/raw/calendar.csv')
    eval_set = read_dataset()
    prices_set = pd.read_csv('../data/raw/sell_prices.csv')

    day_columns = only_days_columns(eval_set)
    train_dataset = eval_set.drop(day_columns[-28:], axis=1)
    val_dataset = eval_set.drop(day_columns[:-28], axis=1)
    metric = WRMSSE(train_dataset, val_dataset, prices_set, levels=LEVELS[:2])

    predict = eval_set.copy()
    predict[metric.validate_days] = (np.random.random(predict[metric.validate_days].shape) * 10).astype('int')
    s = metric.score(predict)
