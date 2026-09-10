"""Snapshot installed core dependencies, without unrelated optional packages.

Run in the environment that passed validation; this records, not resolves,
versions. The resulting lock targets this Python/platform (see its header).
"""
import importlib.metadata as metadata
from pathlib import Path
import platform
import re

from packaging.requirements import Requirement

root = Path(__file__).resolve().parents[1]
pending = [Requirement(line.split('#')[0].strip()) for line in (root / 'requirements.txt').read_text().splitlines()
           if line.strip() and not line.lstrip().startswith('#')]
resolved = {}
while pending:
    requirement = pending.pop()
    if requirement.marker and not requirement.marker.evaluate():
        continue
    key = re.sub(r'[-_.]+', '-', requirement.name).lower()
    dist = metadata.distribution(requirement.name)
    if dist.version not in requirement.specifier:
        raise ValueError(f'{requirement} is not satisfied by {dist.version}')
    if key in resolved:
        continue
    resolved[key] = dist.version
    pending.extend(Requirement(item) for item in (dist.requires or []))
header = f'# Installed core dependency snapshot; Python {platform.python_version()}, {platform.system()} {platform.machine()}.\n'
header += '# Regenerate after validated upgrades: .venv/bin/python scripts/lock_requirements.py\n'
(root / 'requirements.lock').write_text(header + ''.join(f'{name}=={version}\n' for name, version in sorted(resolved.items())))
print(f'Locked {len(resolved)} core and transitive dependencies')
