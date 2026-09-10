"""Attach each existing cluster-experiment bundle's own checksummed mapping."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from ml.pol4.artifacts import sha256_file, verify_checksums, write_json_atomic
from ml.pol4.clustering import ClusterPlan
from ml.pol4.cluster_pipeline import write_clusters_csv
from ml.pol4.config import Pol4Config
from ml.pol4.loader import load_pol4

config = Pol4Config()
data = load_pol4(config)
plan = ClusterPlan.fit(data, config.cutoff, config)
root = config.artifacts_dir.parent / 'pol4_cluster'
summary = json.loads((root / 'run_summary.json').read_text())
for arm, height in [('city', 0), ('clustered', summary['clustered_cut_height'])]:
    directory = root / f'model_bundle_{arm}'
    manifest = json.loads((directory / 'manifest.json').read_text())
    verify_checksums(directory, manifest['files'])
    assignment = plan.assign(height)
    if assignment.n_groups != summary[f'{arm}_model_bundle']['panel_rows']:
        raise ValueError('Assignment does not match the stored model')
    write_clusters_csv(assignment, data, directory / 'clusters.csv')
    manifest['files']['clusters.csv'] = sha256_file(directory / 'clusters.csv')
    manifest['bundle_digest'] = hashlib.sha256('\n'.join(f'{name}:{manifest["files"][name]}' for name in sorted(manifest['files'])).encode()).hexdigest()
    write_json_atomic(manifest, directory / 'manifest.json')
    summary[f'{arm}_model_bundle']['bundle_digest'] = manifest['bundle_digest']
    summary[f'{arm}_model_bundle']['requires'] = f'{directory.name}/clusters.csv - checksummed assignment for this arm'
summary['selected_arm']['sha256'] = sha256_file(root / 'results_selected.csv')
summary['selected_arm']['bundle_digest'] = summary['city_model_bundle']['bundle_digest']
write_json_atomic(summary, root / 'run_summary.json')
print('Both bundle assignments verified and checksummed')
