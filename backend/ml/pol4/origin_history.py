"""Rolling completed-demand statistics indexed by each row's forecast origin.

No forward filling from a later date: origins before the first completed
check-in receive a zero prior. Dense dates preserve structural zero demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import EPS, WEEKEND_DAYS
from .loader import CITY, CHECKIN, SEARCHES


class OriginHistory:
    def __init__(self, data, cutoff, config):
        history = data.history_before(cutoff)
        self.tables = {}
        if history.empty:
            return
        dates = pd.date_range(history[CHECKIN].min(), cutoff)
        totals = history.groupby([CITY, CHECKIN])[SEARCHES].sum()
        window = config.city_history_days
        for city in data.city_codes:
            demand = totals.reindex(
                pd.MultiIndex.from_product([[city], dates], names=[CITY, CHECKIN]),
                fill_value=0,
            )
            demand = pd.Series(demand.to_numpy(dtype=float), index=dates)
            rolling = demand.rolling(window, min_periods=1)
            table = pd.DataFrame({
                "city_hist_mean": rolling.mean(),
                "city_hist_median": rolling.median(),
                "city_hist_p75": rolling.quantile(.75),
                "city_hist_p90": rolling.quantile(.90),
                "city_hist_std": rolling.std().fillna(0),
            })
            table["city_volatility"] = table.city_hist_std / (table.city_hist_mean + EPS)
            weekend = np.isin(dates.dayofweek, WEEKEND_DAYS)
            weekend_mean = demand.where(weekend).rolling(window, min_periods=1).mean()
            weekday_mean = demand.where(~weekend).rolling(window, min_periods=1).mean()
            table["city_weekend_ratio"] = (weekend_mean / (weekday_mean + EPS)).fillna(1)
            for weekday in range(7):
                table[f"weekday_{weekday}"] = demand.where(
                    dates.dayofweek == weekday
                ).rolling(window, min_periods=1).mean().fillna(0)
            self.tables[int(city)] = table

    def block(self, cities, checkins, horizon, cutoff):
        names = ["city_hist_mean", "city_hist_median", "city_hist_p75", "city_hist_p90",
                 "city_hist_std", "city_volatility", "city_weekend_ratio", "city_weekday_mean"]
        out = {name: np.zeros(len(cities)) for name in names}
        origins = pd.DatetimeIndex(checkins) - pd.Timedelta(days=horizon)
        origins = pd.DatetimeIndex(np.minimum(origins.to_numpy(), pd.Timestamp(cutoff).to_datetime64()))
        weekdays = pd.DatetimeIndex(checkins).dayofweek.to_numpy()
        for city in np.unique(cities):
            if int(city) not in self.tables:
                continue
            rows = np.flatnonzero(cities == city)
            values = self.tables[int(city)].reindex(origins[rows]).fillna(0)
            for name in names[:-1]:
                out[name][rows] = values[name].to_numpy()
            means = values[[f"weekday_{day}" for day in range(7)]].to_numpy()
            out["city_weekday_mean"][rows] = means[np.arange(len(rows)), weekdays[rows]]
        return out
