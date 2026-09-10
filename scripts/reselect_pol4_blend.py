"""Apply the practical-gain guard to saved evidence without retraining.

Retains the original decision as history and verifies the city model's files
before redirecting the selected export. The competition submission is separate.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from ml.pol4.artifacts import sha256_file, verify_checksums, write_json_atomic
from ml.pol4.cluster_pipeline import select_blend
from ml.pol4.config import Pol4Config

root = Pol4Config().artifacts_dir.parent / 'pol4_cluster'
path = root / 'run_summary.json'
summary = json.loads(path.read_text())
choice = select_blend(summary['sweep']['blends'], unclustered_wape=summary['selection']['unclustered_wape'])
if choice is not None:
    raise ValueError('A candidate passes; persist and validate its complete source bundle before selecting it')
manifest = json.loads((root / 'model_bundle_city' / 'manifest.json').read_text())
verify_checksums(root / 'model_bundle_city', manifest['files'])
if summary['selected_arm']['arm'] == 'blended':
    summary['previous_selection'] = {"selected_arm": summary['selected_arm'], 'blend_selection': summary['blend_selection']}
    (root / 'results_selected.csv').write_bytes((root / 'results_city.csv').read_bytes())
    summary['blend_selection'] = None
    summary['blend_policy'] = {'min_relative_gain': .01, 'status': 'rejected',
                              'reason': 'Measured gain is below the predeclared practical margin; no significance claim'}
    summary['selected_arm'] = {**summary['selected_arm'], 'arm': 'city',
                               'path': 'artifacts/pol4_cluster/results_selected.csv',
                               'source': 'artifacts/pol4_cluster/results_city.csv',
                               'sha256': sha256_file(root / 'results_selected.csv'),
                               'bundle_digest': manifest['bundle_digest']}
    summary['selection_revision'] = 'Post-audit policy applied to existing sweep; no model was retrained'
    write_json_atomic(summary, path)
print(summary['selected_arm']['arm'])
