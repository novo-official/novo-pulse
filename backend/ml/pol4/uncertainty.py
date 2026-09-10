"""Experimental empirical residual bands with forward-only coverage evaluation.

Repeated cities and regime shifts violate exchangeability. Nominal coverage is
an evaluation target, never a guarantee. Bands belong to the named challenger;
they are not attached to the independently calibrated competition submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_json_atomic
from .config import Pol4Config


def residual_bands(history, current, *, coverage=.8, min_support=100):
    if not 0 < coverage < 1 or min_support < 1:
        raise ValueError('Invalid coverage or minimum support')
    output = current.copy()
    output['lower'] = np.nan
    output['upper'] = np.nan
    output['support'] = 0
    output['band_source'] = 'unavailable'
    if history.empty:
        return output
    history = history.copy()
    history['residual'] = abs(history.actual - history.prediction) / np.maximum(history.prediction, 1)
    for (city, bucket), indices in output.groupby(['city_code', 'horizon_bucket']).groups.items():
        city_rows = history[(history.city_code == city) & (history.horizon_bucket == bucket)]
        bucket_rows = history[history.horizon_bucket == bucket]
        source, rows = ('city/horizon', city_rows) if len(city_rows) >= min_support else ('horizon', bucket_rows)
        if len(rows) < min_support:
            source, rows = 'global', history
        if len(rows) < min_support:
            continue
        values = np.sort(rows.residual.to_numpy())
        rank = int(np.ceil((len(values) + 1) * coverage))
        if rank > len(values):
            continue
        radius = values[rank - 1] * np.maximum(output.loc[indices, 'prediction'], 1)
        output.loc[indices, 'lower'] = np.maximum(output.loc[indices, 'observed'], output.loc[indices, 'prediction'] - radius)
        output.loc[indices, 'upper'] = np.maximum(output.loc[indices, 'observed'], output.loc[indices, 'prediction'] + radius)
        output.loc[indices, 'support'] = len(values)
        output.loc[indices, 'band_source'] = source
    return output


def evaluate_forward(frame, target_days=30, coverage=.8):
    pieces = []
    reports = []
    for cutoff, current in frame.groupby('fold_cutoff', sort=True):
        closed = pd.to_datetime(frame.fold_cutoff) + pd.Timedelta(days=target_days) <= pd.Timestamp(cutoff)
        bands = residual_bands(frame.loc[closed], current, coverage=coverage)
        valid = bands.lower.notna()
        reports.append({'cutoff': cutoff, 'eligible_rows': int(valid.sum()), 'total_rows': len(bands),
                        'observed_coverage': float(((bands.actual >= bands.lower) & (bands.actual <= bands.upper))[valid].mean()) if valid.any() else None,
                        'mean_width': float((bands.upper-bands.lower)[valid].mean()) if valid.any() else None,
                        'history_folds': sorted(frame.loc[closed, 'fold_cutoff'].unique().tolist())})
        pieces.append(bands)
    return pd.concat(pieces, ignore_index=True), reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Pol4Config().artifacts_dir.parent / 'pol4_jury')
    args = parser.parse_args()
    path = args.directory / 'city_calibrated.parquet'
    frame = pd.read_parquet(path)
    bands, reports = evaluate_forward(frame)
    bands.to_parquet(args.directory / 'city_interval_backtest.parquet', index=False)
    # A separately named challenger export, never added to results.csv.
    from .aggregate import aggregate_data
    from .champion import FittedChampion
    from .clustering import ClusterPlan
    from .loader import load_pol4
    from .calibration import Calibrator
    config = Pol4Config()
    data = load_pol4(config)
    model_dir = config.artifacts_dir.parent / 'pol4_cluster' / 'model_bundle_city'
    manifest = json.loads((model_dir / 'manifest.json').read_text())
    contract = json.loads((args.directory / 'contract.json').read_text())
    if manifest['input_digest'] != contract['input_digest']:
        raise ValueError('Challenger bundle and interval evidence use different inputs')
    model = FittedChampion.load(model_dir)
    if model.spec.params.get('n_estimators') != contract['trees']:
        raise ValueError('Challenger bundle has a different tree budget')
    raw = pd.concat([pd.read_parquet(args.directory / f'city_{cutoff}.parquet')
                     for cutoff in contract['cutoffs']], ignore_index=True)
    raw = raw.loc[pd.to_datetime(raw.fold_cutoff) + pd.Timedelta(days=config.target_days) <= config.cutoff]
    calibrator = Calibrator.fit(raw.rename(columns={'prediction': 'predicted_demand'}),
                                'guarded_bias_horizon', config)
    identity = ClusterPlan.fit(data, config.cutoff, config).assign(0)
    forecast = model.predict(config.target_dates(), aggregate_data(data, identity))
    forecast['prediction'] = calibrator.apply(forecast)
    forecast['horizon_bucket'] = forecast.horizon.map(
        lambda h: next(f'{a}-{b}' for a,b in config.horizon_buckets if a <= h <= b))
    history = frame.loc[pd.to_datetime(frame.fold_cutoff) + pd.Timedelta(days=config.target_days) <= config.cutoff]
    target_bands = residual_bands(history, forecast)
    target_path = args.directory / 'city_challenger_intervals.csv'
    target_bands[['city_code','checkin','horizon','observed','prediction','lower','upper',
                  'support','band_source']].to_csv(target_path, index=False)
    write_json_atomic({'status': 'experimental challenger diagnostic', 'nominal_coverage': .8,
                       'source': path.name, 'source_sha256': sha256_file(path), 'folds': reports,
                       'target_export': target_path.name, 'target_sha256': sha256_file(target_path),
                       'bundle_digest': manifest['bundle_digest'],
                       'limitations': ['No bands on the first fold: no earlier out-of-fold errors',
                                       'Dependent rows and regime shifts: no coverage guarantee',
                                       'Not deployed on the submitted model']}, args.directory / 'uncertainty.json')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
