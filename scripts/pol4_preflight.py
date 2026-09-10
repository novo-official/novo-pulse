"""Read-only readiness checks, with one generated JSON report."""
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

import numpy as np
import pandas as pd
from apps.pol4.services import jury_evidence
from ml.pol4.artifacts import sha256_file, verify_checksums, write_json_atomic
from ml.pol4.champion import FittedChampion
from ml.pol4.config import Pol4Config

config = Pol4Config()
root = config.artifacts_dir.parent.parent
checks = {}
failures = {}

def check(name, fn):
    try:
        value = fn()
        if value is False:
            raise ValueError('Check returned false')
        checks[name] = value if value is not None else True
    except Exception as exc:
        failures[name] = str(exc)

def dependencies():
    mismatches = []
    for line in (root / 'requirements.lock').read_text().splitlines():
        if line and not line.startswith('#'):
            name, version = line.split('==')
            actual = importlib.metadata.version(name)
            if actual != version:
                mismatches.append(f'{name}: expected {version}, found {actual}')
    if mismatches:
        raise ValueError('; '.join(mismatches))
    return True

def submission():
    frame = pd.read_csv(config.artifacts_dir / 'results.csv')
    assert len(frame) == 9630 and frame.cluster_code.nunique() == 321 and frame.checkin.nunique() == 30
    assert not frame.duplicated(['cluster_code','checkin']).any()
    assert np.isfinite(frame.predicted_demand).all() and (frame.predicted_demand >= 0).all()
    assert set(frame.checkin) == {d.date().isoformat() for d in config.target_dates()}
    return {'rows': len(frame), 'sha256': sha256_file(config.artifacts_dir / 'results.csv')}

def recovered():
    report = json.loads((config.artifacts_dir / 'trainset_recovery.json').read_text())
    assert report['rows'] == report['recorded_rows'] == 2655312
    assert all(report['file_hash_matches'].values())
    manifest = json.loads((config.artifacts_dir / 'trainset_manifest.json').read_text())
    verify_checksums(config.artifacts_dir / 'trainset', manifest['files'])
    return True

def bundle(path):
    loaded = FittedChampion.load(path)
    return {'features': loaded.spec.describe()['n_features'], 'rows': loaded.training_rows}

def selection():
    path = config.artifacts_dir.parent / 'pol4_cluster'
    summary = json.loads((path / 'run_summary.json').read_text())
    assert summary['selected_arm']['arm'] == 'city'
    assert sha256_file(path / 'results_selected.csv') == sha256_file(path / 'results_city.csv')
    for arm, expected in [('city',321),('clustered',109)]:
        mapping = pd.read_csv(path / f'model_bundle_{arm}' / 'clusters.csv')
        assert mapping.city_code.nunique() == 321
        assert mapping.cluster_code.nunique() == expected
    return 'city; blend rejected; each model has its own checksummed assignment'

check('locked_environment', dependencies)
check('submission', submission)
check('recovered_trainset', recovered)
check('champion_reload', lambda: bundle(config.artifacts_dir / 'model_bundle'))
check('city_challenger_reload', lambda: bundle(config.artifacts_dir.parent / 'pol4_cluster/model_bundle_city'))
check('cluster_challenger_reload', lambda: bundle(config.artifacts_dir.parent / 'pol4_cluster/model_bundle_clustered'))
check('cluster_selection', selection)
check('fresh_evidence', lambda: bool(jury_evidence()))
report = {'passed': not failures, 'checks': checks, 'failures': failures}
write_json_atomic(report, config.artifacts_dir / 'jury_preflight.json')
print(json.dumps(report, indent=2))
raise SystemExit(1 if failures else 0)
