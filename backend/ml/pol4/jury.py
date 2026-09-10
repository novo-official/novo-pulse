"""Build the judge-facing evidence and business decision packet from artifacts.

Every displayed metric is derived, every source is checksummed, and historical
shock diagnostics never enter forecast features or the submission.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_json_atomic
from .config import Pol4Config
from .validation import _pool_fold_scores

SHOCK = {"start": "2025-06-13", "end": "2025-06-24", "fold": "2025-05-21",
         "label": "June 2025 Iran–Israel war",
         "source": "https://dppa.un.org/en/node/101504",
         "interpretation": "Temporal association; causal attribution is not established. Exclusion is diagnostic only."}


def shock_sensitivity(folds):
    pooled = _pool_fold_scores(folds.values())
    hard = folds.get(SHOCK['fold'])
    without = _pool_fold_scores(v for k, v in folds.items() if k != SHOCK['fold'])
    errors = sum(v['wape'] * v['actual_total'] for v in folds.values())
    return {**SHOCK, 'all_folds_wape': pooled.get('wape'),
            'without_fold_wape': without.get('wape') if hard else None,
            'error_share': hard['wape'] * hard['actual_total'] / errors if hard and errors else None,
            'folds': len(folds), 'excluded_folds': 1 if hard else 0}


def decision_queue(named, momentum):
    """Search-interest review queue; no spending or supply recommendations."""
    totals = named.groupby(['city_code', 'city', 'province'], as_index=False)[
        ['observed_so_far', 'predicted_demand', 'predicted_remaining']].sum()
    total = totals.predicted_demand.sum()
    ratios = momentum.set_index('city_code').pickup_ratio
    totals['forecast_share'] = totals.predicted_demand / total if total else 0
    totals['observed_share'] = np.divide(totals.observed_so_far, totals.predicted_demand,
                                         out=np.zeros(len(totals)), where=totals.predicted_demand > 0)
    totals['pickup_ratio'] = totals.city_code.map(ratios)
    peaks = named.loc[named.groupby('city_code').predicted_demand.idxmax()].set_index('city_code').checkin
    totals['peak_checkin'] = totals.city_code.map(peaks)
    totals['owner'] = 'Destination growth analyst'
    totals['action'] = 'Review destination content and campaign calendar'
    totals['required_before_spend'] = 'Join conversion, available inventory, margin and campaign cost; run a controlled pilot'
    totals['evidence_status'] = np.where(totals.observed_so_far == 0,
                                         'No observed searches: manual review',
                                         'Historical forecast; analyst review required')
    totals['priority_reason'] = 'Ranked by forecast search volume; pickup ratio is supporting evidence'
    return totals.sort_values(['predicted_demand', 'city_code'], ascending=[False, True]).reset_index(drop=True)


def build(config=None):
    config = config or Pol4Config()
    directory = config.artifacts_dir
    sources = {}
    def read(relative):
        path = directory / relative
        sources[relative] = sha256_file(path)
        return json.loads(path.read_text())
    metrics = read('backtest_metrics_phase2.json')
    summary = read('run_summary.json')
    validation = read('validation_audit.json')
    named_path = directory / 'results_named.csv'
    sources['results_named.csv'] = sha256_file(named_path)
    named = pd.read_csv(named_path)
    sources['results.csv'] = sha256_file(directory / 'results.csv')
    submission = pd.read_csv(directory / 'results.csv')
    expected = named[['city_code', 'checkin', 'predicted_demand']].rename(columns={'city_code': 'cluster_code'})
    pd.testing.assert_frame_equal(submission.sort_values(['cluster_code','checkin']).reset_index(drop=True),
                                  expected.sort_values(['cluster_code','checkin']).reset_index(drop=True), check_dtype=False)
    sources['city_momentum.parquet'] = sha256_file(directory / 'city_momentum.parquet')
    queue = decision_queue(named, pd.read_parquet(directory / 'city_momentum.parquet'))
    queue.to_csv(directory / 'decision_queue.csv', index=False)
    cluster_path = directory.parent / 'pol4_cluster' / 'run_summary.json'
    clustering = None
    if cluster_path.exists():
        cluster = read('../pol4_cluster/run_summary.json')
        clustering = {'cutoffs': cluster['sweep'].get('cutoffs', []),
                      'levels': cluster['sweep'].get('gain_decomposition', []),
                      'selected_arm': cluster['selected_arm']['arm'],
                      'warning': 'Three-fold uncalibrated experiment. Compare levels within this sweep only.',
                      'blend_min_relative_gain': config.blend_min_relative_gain}
    comparison_path = directory.parent / 'pol4_jury' / 'comparison.json'
    experiments = read('../pol4_jury/comparison.json') if comparison_path.exists() else None
    history_reports = {}
    history_contracts = {}
    for arm, folder in [('champion66', 'champion'), ('origin66', 'origin')]:
        relative = f'../pol4_history66/{folder}/comparison.json'
        if (directory / relative).exists():
            report = read(relative)
            history_reports[arm] = report['arms'][arm]
            history_contracts[arm] = report['contract']
    if experiments and len(history_reports) == 2:
        experiments['arms'].update(history_reports)
        experiments['supplemental_contracts'] = history_contracts
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'cutoff': summary['cutoff'], 'mode': 'historical competition snapshot',
        'target_window': summary['target_window'],
        'scope': {'cities': int(named.city_code.nunique()), 'provinces': sorted(named.province.unique()),
                  'dates': int(named.checkin.nunique()), 'target': 'search interest'},
        'performance': {'champion': metrics['champion']['pooled'], 'baseline': metrics['baseline']['pooled'],
                        'relative_improvement': 1 - metrics['champion']['pooled']['wape'] / metrics['baseline']['pooled']['wape'],
                        'confidence_interval': validation['reported_selected']['confidence_interval'],
                        'prequential_wape': validation['prequential_selection']['pooled']['wape'],
                        'folds': [{'cutoff': k, 'champion': v['wape'], 'baseline': metrics['baseline']['folds'][k]['wape'],
                                   'shock_overlap': k == SHOCK['fold']} for k,v in metrics['champion']['folds'].items()]},
        'shock': shock_sensitivity(metrics['champion']['folds']),
        'raw_shock': shock_sensitivity(metrics['champion_raw']['folds']),
        'clustering': clustering, 'experiments': experiments,
        'event_study': read('shock_event_study.json') if (directory / 'shock_event_study.json').exists() else None,
        'uncertainty': read('../pol4_jury/uncertainty.json') if (directory.parent / 'pol4_jury/uncertainty.json').exists() else None,
        'submission': {'rows': len(submission), 'finite': bool(np.isfinite(submission.predicted_demand).all()),
                       'nonnegative': bool((submission.predicted_demand >= 0).all()),
                       'observed_floor': bool((named.predicted_demand >= named.observed_so_far).all()),
                       'duplicate_keys': int(submission.duplicated(['cluster_code', 'checkin']).sum()),
                       'sha256': sources['results.csv']},
        'trainset_recovery': read('trainset_recovery.json') if (directory / 'trainset_recovery.json').exists() else None,
        'decisions': queue.to_dict('records'),
        'limits': ['Search interest across seven provinces; bookings, occupancy, revenue and supply gaps are unmeasured.',
                   'The submitted model uses fold-frozen city history; the origin-history ablation is reported separately.',
                   'Five validation folds; model specification was preselected, not nested.',
                   'No source-verified lunar-event features or deployed predictive intervals in the submitted model.',
                   'Business outcomes require a prospective pilot with joined booking and inventory data.'],
        'sources': sources,
    }
    from .pipeline import _json_ready
    payload = _json_ready(payload)
    write_json_atomic(payload, directory / 'jury_evidence.json')
    return payload


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    payload = build()
    print(f"Jury evidence generated: {payload['submission']['rows']} submission rows; {len(payload['decisions'])} city reviews")


if __name__ == '__main__':
    main()
