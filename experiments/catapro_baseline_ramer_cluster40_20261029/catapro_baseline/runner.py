#!/usr/bin/env python3
"""Train the original CataPro KcatModel on an immutable fixed split.

This wrapper changes neither the model nor the loss.  It only adapts the
original CataPro training loop to approved new cluster40, train terminal8+1 merged9
and makes the run reproducible/auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tail_batch import MergeSingletonBatchSampler


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from master_store import MasterStore
_INPUT_STORES = {}

EXPECTED_ROWS = {"train": 21585, "valid": 3037, "test": 3036}
EXPECTED_FEATURE_COLUMNS = 1960


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catapro-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate all files and run one train/valid batch; never evaluate test.",
    )
    return parser.parse_args()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TensorRows(Dataset):
    def __init__(self, values: torch.Tensor):
        self.values = values

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)


def load_split(
    split: str,
    feature_dir: Path,
    split_dir: Path,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, pd.DataFrame, dict[str, Any]]:
    if feature_dir != split_dir:
        raise RuntimeError("Both input directories must point to the audited shared inputs")
    manifest_path = feature_dir / "feature_store.json"
    rowspec_path = feature_dir / "rowspec.json"
    if str(feature_dir) not in _INPUT_STORES:
        store = MasterStore(manifest_path)
        spec = json.loads(rowspec_path.read_text(encoding="utf-8"))
        all_rows = spec["train"] + spec["valid"] + spec["eval"]["test"]
        all_ids = [str(sid) for source, sid in all_rows]
        if any(source != "catapro" for source,sid in all_rows) or len(set(all_ids)) != len(all_ids):
            raise RuntimeError("Repeated or unexpected source row in rowspec")
        if set(all_ids) != set(store.indices):
            raise RuntimeError("Rowspec must cover exactly the shared feature population")
        _INPUT_STORES[str(feature_dir)] = (store, spec)
    store, spec = _INPUT_STORES[str(feature_dir)]
    rows = spec[split] if split != "test" else spec["eval"]["test"]
    assignment_path = Path(__file__).resolve().parents[1] / "assignments.csv"
    if sha256(assignment_path) != spec["meta"]["assignments_sha256"]:
        raise RuntimeError("Frozen assignments changed")
    assignment_frame = pd.read_csv(assignment_path, dtype=str)
    membership = dict(zip(assignment_frame.balance_row_uid.str.removeprefix("catapro:"), assignment_frame.new_split))
    feature_ids, feature_rows, labels = [], [], []
    for source, sid in rows:
        row = store.get(source, sid)
        if membership[str(sid)] != split:
            raise RuntimeError("Wrong frozen cluster40 membership for " + split)
        feature_ids.append(str(sid))
        feature_rows.append(row["features"])
        labels.append(row["label"])
    values = np.concatenate((np.stack(feature_rows), np.asarray(labels,dtype=np.float32)[:,None]), axis=1)
    if values.shape != (EXPECTED_ROWS[split], EXPECTED_FEATURE_COLUMNS):
        raise RuntimeError("Wrong complete author population for " + split)
    if len(set(feature_ids)) != len(feature_ids) or not np.isfinite(values).all():
        raise RuntimeError("Repeated ID or nonfinite author inputs")
    metadata = pd.DataFrame(dict(row_uid=feature_ids, Type="kcat", log_Kcat=values[:,-1],new_split=split))
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorRows(torch.from_numpy(values))
    if split == "train":
        sampler = MergeSingletonBatchSampler(RandomSampler(dataset, generator=generator), batch_size=batch_size, drop_last=False)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, worker_init_fn=seed_worker, generator=generator, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, worker_init_fn=seed_worker, generator=generator, pin_memory=True, drop_last=False)
    audit = {
        "rows": len(metadata),
        "feature_columns": values.shape[1],
        "feature_sha256": store.manifest["features_npy_sha256"],
        "metadata_sha256": store.manifest["rows_csv_sha256"],
        "feature_manifest_sha256": sha256(manifest_path),
        "rowspec_sha256": sha256(rowspec_path),
        "first_row_uid": feature_ids[0],
        "last_row_uid": feature_ids[-1],
    }
    return loader, metadata, audit


def metric_dict(train_utils: Any, labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    pcc, scc, r2, rmse = train_utils.evaluate(labels, predictions)
    mae = np.mean(np.abs(predictions - labels))
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "PCC": float(pcc),
        "SCC": float(scc),
    }


def eval_array(values: np.ndarray) -> dict[str, float]:
    names = ("PCC", "SCC", "R2", "RMSE", "batch_average_RMSELoss")
    return {name: float(value) for name, value in zip(names, values)}


def main() -> None:
    args = parse_args()
    expected = (668, 8, 1e-5, 0.0, 150, 20, 0.001, 8)
    received = (args.seed,args.batch_size,args.lr,args.drop_rate,args.epochs,args.patience,args.min_delta,args.num_workers)
    if received != expected:
        raise RuntimeError("Outside approved legacy150 settings")
    root = args.catapro_root.resolve()
    split_dir = args.split_dir.resolve()
    feature_dir = args.feature_dir.resolve()
    output = args.output_dir.resolve()

    if args.seed != 668:
        raise RuntimeError(f"This prepared experiment requires seed 668, received {args.seed}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("A CUDA device is required for this CataPro run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)

    source_model = root / "training" / "kcat" / "model.py"
    source_utils = root / "training" / "kcat" / "utils.py"
    for path in (source_model, source_utils):
        if not path.is_file():
            raise FileNotFoundError(path)

    if sha256(source_model) != 'ad25f4136e60ed31edd0ad75b5dcc5b55da686ffad9ce89e657902587b5f41f5' or sha256(source_utils) != '1c5dfb08259a90eee896ec9ca1b3fcbb8dfe641b53e85f750b64f7ddaeaa762d':
        raise RuntimeError("Author model/utils bytes changed")

    set_seed(args.seed)
    model_module = load_module("catapro_kcat_model", source_model)
    train_utils = load_module("catapro_kcat_utils", source_utils)

    loaders: dict[str, DataLoader] = {}
    metadata: dict[str, pd.DataFrame] = {}
    input_audit: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        loaders[split], metadata[split], input_audit[split] = load_split(
            split,
            feature_dir,
            split_dir,
            args.batch_size,
            args.num_workers,
            args.seed,
        )

    protocol = {
        "protocol": "CataPro new cluster40; log10(kcat s^-1); training terminal8+1 merged9; validation unchanged",
        "test_policy": "test is evaluated only after validation checkpoint selection",
        "architecture": "unmodified CataPro training/kcat/model.py::KcatModel",
        "loss_and_training_functions": "unmodified CataPro training/kcat/utils.py",
        "parameters": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "drop_rate": args.drop_rate,
            "max_epochs": args.epochs,
            "early_stopping": True,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "optimizer": "Adam(betas=(0.9,0.999), weight_decay=0.01)",
            "checkpoint_selection": "minimum validation batch-average RMSELoss under original EarlyStopping.check",
            "num_workers": args.num_workers,
            "device": args.device,
        },
        "source_sha256": {
            "model.py": sha256(source_model),
            "utils.py": sha256(source_utils),
            "runner": sha256(Path(__file__).resolve()),
        },
        "inputs": input_audit,
    }
    write_json_atomic(output / "protocol.json", protocol)
    write_json_atomic(
        output / "status.json",
        {"state": "preflight" if args.preflight else "training", "started_unix": time.time()},
    )

    model = model_module.KcatModel(rate=args.drop_rate, device=args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    if args.preflight:
        train_batch = next(iter(loaders["train"]))
        valid_batch = next(iter(loaders["valid"]))
        train_values = train_utils.run_a_training_epoch(
            model, [train_batch, torch.stack([loaders["train"].dataset[i] for i in range(len(loaders["train"].dataset)-9, len(loaders["train"].dataset))])], optimizer, args.device
        )
        _, valid_labels, valid_values = train_utils.run_an_eval_epoch(
            model, [valid_batch], args.device
        )
        if not np.isfinite(train_values).all() or not np.isfinite(valid_values).all():
            raise RuntimeError("Nonfinite preflight metrics")
        for name, value in model.state_dict().items():
            if not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite model state: " + name)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise RuntimeError("Nonfinite optimizer state: " + key)
        result = {
            "state": "preflight_complete",
            "tail_batch_rows": 9,
            "model_optimizer_finite": True,
            "test_evaluated": False,
            "train_batch_rows": int(len(train_batch)),
            "valid_batch_rows": int(len(valid_labels)),
            "train_batch_metrics": eval_array(train_values),
            "valid_batch_metrics": eval_array(valid_values),
        }
        write_json_atomic(output / "preflight.json", result)
        write_json_atomic(output / "status.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return

    stop = train_utils.EarlyStopping(args.patience, args.min_delta)
    best_path = output / "checkpoints" / "bestmodel.pth"
    records: list[np.ndarray] = []
    history: list[dict[str, Any]] = []
    best_epoch: int | None = None

    previous_cwd = Path.cwd()
    os.chdir(output)
    try:
        for epoch in range(args.epochs):
            train_values = train_utils.run_a_training_epoch(
                model, loaders["train"], optimizer, args.device
            )
            _, _, valid_values = train_utils.run_an_eval_epoch(
                model, loaders["valid"], args.device
            )
            record = np.concatenate([np.array([epoch]), train_values, valid_values])
            records.append(record)
            train_utils.write_logfile(epoch, records, "training_log.csv")
            is_best, stopping = stop.check(epoch, valid_values[-1])
            if is_best:
                best_epoch = epoch
                tmp_checkpoint = best_path.with_suffix(".pth.tmp")
                torch.save(model.state_dict(), tmp_checkpoint)
                os.replace(tmp_checkpoint, best_path)
            epoch_record = {
                "epoch": epoch,
                "train": eval_array(train_values),
                "valid": eval_array(valid_values),
                "is_best": bool(is_best),
                "early_stopping": bool(stopping),
            }
            history.append(epoch_record)
            write_json_atomic(output / "history.json", history)
            write_json_atomic(
                output / "status.json",
                {
                    "state": "training",
                    "last_completed_epoch": epoch,
                    "epochs_completed": len(history),
                    "best_epoch": best_epoch,
                    "early_stopping_triggered": bool(stopping),
                },
            )
            print(json.dumps(epoch_record), flush=True)
            if stopping:
                break
    finally:
        os.chdir(previous_cwd)

    if best_epoch is None or not best_path.is_file():
        raise RuntimeError("No best checkpoint was saved")

    best_model = model_module.KcatModel(rate=args.drop_rate, device=args.device)
    state_dict = torch.load(best_path, map_location=args.device)
    load_result = best_model.load_state_dict(state_dict, strict=True)
    if not all(torch.isfinite(v).all() for v in best_model.state_dict().values()):
        raise RuntimeError("Nonfinite best checkpoint")
    split_results: dict[str, Any] = {}
    for split in ("valid", "test"):
        predictions, labels, values = train_utils.run_an_eval_epoch(
            best_model, loaders[split], args.device
        )
        if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
            raise RuntimeError("Nonfinite final predictions")
        prediction_frame = metadata[split][["row_uid", "Type", "log_Kcat"]].copy()
        prediction_frame["prediction"] = predictions
        prediction_frame.to_csv(output / f"{split}_predictions.csv", index=False)
        split_results[split] = {
            "rows": int(len(labels)),
            "metrics": metric_dict(train_utils, labels, predictions),
            "reported_batch_average_RMSELoss": float(values[-1]),
        }

    summary = {
        **protocol,
        "best_epoch_zero_based": best_epoch,
        "epochs_ran": len(history),
        "early_stopping_triggered": len(history) < args.epochs,
        "strict_load_missing_keys": list(load_result.missing_keys),
        "strict_load_unexpected_keys": list(load_result.unexpected_keys),
        "results": split_results,
    }
    write_json_atomic(output / "summary.json", summary)
    write_json_atomic(
        output / "status.json",
        {
            "state": "complete",
            "best_epoch_zero_based": best_epoch,
            "epochs_ran": len(history),
            "test_rows": split_results["test"]["rows"],
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
