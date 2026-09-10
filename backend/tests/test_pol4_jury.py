from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml.pol4 import load_pol4
from ml.pol4.features import CityHistory
from ml.pol4.origin_history import OriginHistory
from ml.pol4.cluster_pipeline import select_blend
from tests.pol4_fixtures import write_dataset


def test_origin_history_matches_independent_refits_and_ignores_future(tmp_path):
    config = write_dataset(tmp_path)
    config.city_history_days = 60
    data = load_pol4(config)
    cache = OriginHistory(data, config.cutoff, config)
    for origin in (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-04-12'), config.cutoff):
        checkins = pd.DatetimeIndex([origin + pd.Timedelta(days=7)] * len(data.city_codes))
        expected = CityHistory.fit(data, origin, config).block(data.city_codes, checkins.dayofweek)
        actual = cache.block(data.city_codes, checkins, 7, config.cutoff)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-8)
    origin = pd.Timestamp('2024-04-12')
    poisoned = data.search.copy()
    poisoned.loc[poisoned.checkin > origin, 'search_count'] *= 10000
    altered = OriginHistory(replace(data, search=poisoned), config.cutoff, config)
    checkins = pd.DatetimeIndex([origin + pd.Timedelta(days=7)] * len(data.city_codes))
    before = cache.block(data.city_codes, checkins, 7, config.cutoff)
    after = altered.block(data.city_codes, checkins, 7, config.cutoff)
    for key in before:
        np.testing.assert_allclose(before[key], after[key], atol=1e-8)
    cold = cache.block(data.city_codes, pd.DatetimeIndex(['2023-01-01'] * len(data.city_codes)), 7, config.cutoff)
    assert all((values == 0).all() for values in cold.values())


def test_blend_rejects_noise_and_invalid_scores():
    def candidate(wape):
        return {'shrinkage': 10000, 'pooled': {'wape': wape}}
    assert select_blend([candidate(.120096)], unclustered_wape=.120168) is None
    assert select_blend([candidate(np.nan)], unclustered_wape=.12) is None
    assert select_blend([candidate(.1)], unclustered_wape=.12) is not None
    assert select_blend([candidate(.1)], unclustered_wape=0) is None
    with pytest.raises(ValueError):
        select_blend([], unclustered_wape=.12, min_relative_gain=-1)


def test_interval_coverage_uses_only_closed_earlier_folds():
    from ml.pol4.uncertainty import evaluate_forward
    frame = pd.DataFrame({'city_code': [1]*360, 'horizon_bucket': ['1-3']*360,
                          'prediction': [100.]*360, 'observed': [80.]*360,
                          'actual': [110.]*360,
                          'fold_cutoff': ['2025-01-01']*120 + ['2025-01-15']*120 + ['2025-03-01']*120})
    before, report = evaluate_forward(frame)
    assert report[0]['eligible_rows'] == report[1]['eligible_rows'] == 0
    assert report[2]['history_folds'] == ['2025-01-01', '2025-01-15']
    frame.loc[frame.fold_cutoff == '2025-03-01', 'actual'] = 1e9
    after, _ = evaluate_forward(frame)
    pd.testing.assert_frame_equal(before[['lower','upper']], after[['lower','upper']])
    valid = after.lower.notna()
    assert (after.loc[valid, 'lower'] >= after.loc[valid, 'observed']).all()


def test_shock_diagnostic_is_demand_weighted():
    from ml.pol4.jury import shock_sensitivity
    folds = {'2025-05-21': {'n':1, 'actual_total':100, 'predicted_total':70, 'wape':.3},
             '2025-10-22': {'n':1, 'actual_total':900, 'predicted_total':810, 'wape':.1}}
    result = shock_sensitivity(folds)
    assert result['all_folds_wape'] == pytest.approx(.12)
    assert result['without_fold_wape'] == pytest.approx(.1)
    assert result['error_share'] == pytest.approx(.25)


def test_stale_or_missing_evidence_is_not_served(client, tmp_path, monkeypatch):
    import json
    from apps.pol4 import services
    from ml.pol4.config import Pol4Config
    from ml.pol4.artifacts import sha256_file
    monkeypatch.setattr(services, 'CONFIG', Pol4Config(artifacts_dir=tmp_path))
    services.clear_cache()
    assert client.get('/api/v1/pol4/jury/').json()['available'] is False
    source = tmp_path / 'test.csv'
    source.write_text('actual\n1\n')
    (tmp_path / 'jury_evidence.json').write_text(json.dumps({'sources': {'test.csv': sha256_file(source)}}))
    assert client.get('/api/v1/pol4/jury/').json()['available'] is True
    source.write_text('actual\n2\n')
    assert client.get('/api/v1/pol4/jury/').json()['available'] is False
    services.clear_cache()
