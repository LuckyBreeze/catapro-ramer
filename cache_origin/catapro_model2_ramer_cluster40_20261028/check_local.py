"""Audit exact old computation plus meaningful new RAMER cache/identity tests."""
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'catapro_residue_query_model2_20261027'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    identity = json.loads((ROOT / 'baseline_identity.json').read_text())
    for name, digest in identity['copied_unchanged'].items():
        assert sha(ROOT / name) == digest == sha(OLD / name), name
    trees = [ast.parse((root / 'train.py').read_text(encoding='utf-8')) for root in (OLD, ROOT)]
    functions = [{n.name: ast.dump(n, include_attributes=False) for n in t.body
                  if isinstance(n, ast.FunctionDef)} for t in trees]
    assert functions[0].keys() == functions[1].keys()
    kept = [name for name in functions[0] if name != 'source_identity']
    for name in kept:
        assert functions[0][name] == functions[1][name], name
    # Complete module equality after replacing only the source-identity function.
    for t in trees:
        t.body = [n for n in t.body if not isinstance(n, ast.FunctionDef) or n.name != 'source_identity']
    assert ast.dump(trees[0]) == ast.dump(trees[1])
    before = (OLD / 'data.py').read_text(encoding='utf-8')
    after = (ROOT / 'data.py').read_text(encoding='utf-8')
    addition = ("    if kind == 'protein':\n"
                "        from ramer_cache import verify_ramer_cache\n"
                "        return verify_ramer_cache(root, items_path)\n")
    assert after.replace(addition, '', 1) == before
    import test_train
    import test_ramer
    suite = unittest.TestLoader().loadTestsFromModule(test_ramer)
    checks = ('test_preflight_updates_discarded_and_model_optimizer_roundtrip',
              'test_terminal_singleton_is_merged_without_dropping_any_row',
              'test_eval_identity_and_batch_mean_loss', 'test_loss_rejects_broadcast_shape',
              'test_real_collator_preflight_retains_both_length_maxima')
    suite.addTests(test_train.TrainChecks(name) for name in checks)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (ROOT / 'local_tests.txt').write_text(stream.getvalue(), encoding='utf-8')
    print(stream.getvalue(), flush=True)
    report = dict(status='passed' if result.wasSuccessful() else 'failed',
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  training_functions_ast_identical=kept,
                  entire_training_computation_ast_identical=True,
                  only_data_change='protein cache verifier; old MolT5/readout/collator unchanged',
                  copied_unchanged=identity['copied_unchanged'],
                  reviewed_sha256={p.name: sha(p) for p in ROOT.glob('*.py')},
                  real_data_training=False, submitted_jobs=0)
    (ROOT / 'local_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not result.wasSuccessful():
        raise RuntimeError('Local RAMER/CPU model2 checks failed')


if __name__ == '__main__':
    main()
