"""Descriptive shock-window study; no causal treatment/control claim."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

import jdatetime
import pandas as pd
from ml.pol4.artifacts import sha256_file, write_json_atomic
from ml.pol4.config import Pol4Config
from ml.pol4.jury import SHOCK

config = Pol4Config()
raw = pd.read_csv(config.search_path, usecols=['log_date', 'checkin', 'search_count'], parse_dates=['log_date','checkin'])
outputs = []
periods = {'pre': ('2025-05-14','2025-06-12'), 'during': ('2025-06-13','2025-06-24'), 'post': ('2025-06-25','2025-07-24')}
for clock in ['log_date', 'checkin']:
    daily = raw.groupby(clock).search_count.sum()
    dates = pd.date_range('2025-05-14','2025-07-24')
    table = pd.DataFrame({'date': dates, 'searches': daily.reindex(dates, fill_value=0).to_numpy()})
    pre = table[table.date.between(*periods['pre'])]
    weekday = pre.groupby(pre.date.dt.dayofweek).searches.mean()
    table['weekday_expected'] = table.date.dt.dayofweek.map(weekday)
    table['weekday_ratio'] = table.searches / table.weekday_expected
    def last_year(stamp):
        jalali = jdatetime.date.fromgregorian(date=stamp.date())
        return pd.Timestamp(jdatetime.date(jalali.year-1, jalali.month, jalali.day).togregorian())
    table['prior_jalali_date'] = table.date.map(last_year)
    table['prior_year_searches'] = table.prior_jalali_date.map(daily)
    table['clock'] = clock
    table['period'] = 'post'
    for name, bounds in periods.items():
        table.loc[table.date.between(*bounds), 'period'] = name
    outputs.append(table)
frame = pd.concat(outputs, ignore_index=True)
path = config.artifacts_dir / 'shock_daily.csv'
frame.to_csv(path, index=False)
summary = []
for (clock, period), rows in frame.groupby(['clock','period']):
    summary.append({'clock': clock, 'period': period, 'days': len(rows),
                    'daily_mean': float(rows.searches.mean()),
                    'weekday_adjusted_ratio': float(rows.searches.sum()/rows.weekday_expected.sum()),
                    'matched_jalali_year_ratio': float(rows.searches.sum()/rows.prior_year_searches.sum())})
write_json_atomic({'event': SHOCK, 'windows': periods, 'summary': summary,
                   'input_sha256': sha256_file(config.search_path), 'daily_sha256': sha256_file(path),
                   'limitations': ['Descriptive window and weekday comparisons, not causal identification',
                                   'Prior-year Jalali matching does not control lunar holidays or secular growth',
                                   'No clean untreated group for a broad regional shock',
                                   'Diagnostic only; never a known-future model feature']},
                  config.artifacts_dir / 'shock_event_study.json')
print(json.dumps(summary, indent=2))
