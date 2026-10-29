"""Audited baseline definitions and immutable, split-aligned residue inputs."""
import ast
import hashlib
import json
import os
from pathlib import Path
import types

import numpy as np
import torch
from torch.utils.data import Dataset

BASELINE_SHA = '28016263a7307d8133a0c1fe609ce935663056dade0a39f29afe9626f1d3a405'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def baseline(path):
    if sha(path) != BASELINE_SHA:
        raise RuntimeError('Historical baseline source hash changed')
    tree = ast.parse(Path(path).read_text())
    # Never execute the baseline command-line entry point on import.
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    module = types.ModuleType('audited_baseline')
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module



def collate(rows):
    lengths = [len(r[1]) for r in rows]
    residues = torch.zeros(len(rows), max(lengths), 2560)
    mask = torch.zeros(len(rows), max(lengths), dtype=torch.bool)
    for i, row in enumerate(rows):
        residues[i, :lengths[i]] = row[1]
        mask[i, :lengths[i]] = True
    return (torch.stack([r[0] for r in rows]), residues, mask,
            torch.stack([r[2] for r in rows]))
