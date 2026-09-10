"""Regenerate the submitted model's training frame, without fitting or changing it."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from ml.pol4.artifacts import build_input_manifest, write_json_atomic
from ml.pol4.champion import FittedChampion
from ml.pol4.config import Pol4Config
from ml.pol4.dataset import build_training_frame, materialize_training_frame
from ml.pol4.loader import load_pol4

config = Pol4Config()
fitted = FittedChampion.load(config.artifacts_dir / 'model_bundle')
# Relocate the old machine's paths, retaining every fitted training parameter.
fitted.config.raw_dir = config.raw_dir
fitted.config.artifacts_dir = config.artifacts_dir
config = fitted.config
data = load_pol4(config)
inputs = build_input_manifest(config, data)
summary = json.loads((config.artifacts_dir / 'run_summary.json').read_text())
expected = summary['champion']['trainset']
if inputs['input_digest'] != expected['input_digest']:
    raise ValueError('Inputs do not match the champion')
train = build_training_frame(data, fitted.cutoff, fitted.spec.groups, fitted.baseline, config)
manifest = materialize_training_frame(train, config.artifacts_dir / 'trainset', fitted.spec.groups,
                                      config, input_digest=inputs['input_digest'])
write_json_atomic(manifest, config.artifacts_dir / 'trainset_manifest.json')
report = {'rows': len(train), 'recorded_rows': expected['rows'],
          'file_hash_matches': {name: digest == expected['files'][name] for name, digest in manifest['files'].items()},
          'input_digest_matches': True,
          'note': 'Rebuilt with original bundle parameters; byte hashes can differ with parquet/library versions.'}
write_json_atomic(report, config.artifacts_dir / 'trainset_recovery.json')
print(json.dumps(report, indent=2))
