"""Ensure new encoder features did not alter the downstream starting model."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
baseline = json.loads((root / 'baseline_identity.json').read_text())
preflight = json.loads((root / 'run_seed668' / 'preflight' / 'preflight.json').read_text())
assert preflight['state'] == 'preflight_complete'
assert preflight['initial_sha256'] == baseline['old_formal_initial_sha256']
assert preflight['model_roundtrip'] and preflight['adam_roundtrip'] and preflight['eval_roundtrip_exact']
print('RAMER model2 GPU preflight and original downstream initialization matched', flush=True)
