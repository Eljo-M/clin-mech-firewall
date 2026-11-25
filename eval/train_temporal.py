"""Fit and select the temporal probes using training and validation charts only."""

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from eval.run import digest, write_json, write_jsonl
from mech_interp.data import build_dataset
from mech_interp.model import ResidualModel
from mech_interp.snapshot import DEFAULT_MODEL, DEFAULT_REVISION, verify_model_source
from mech_interp.temporal_data import build_temporal_dataset
from mech_interp.temporal_probe import (
    BASELINE_SHA256, SELECTION_ORDER, THRESHOLD_RULE, develop_probes, select_model,
)


def train(folder, args):
    source = args.original_probe.read_bytes()
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA256:
        raise ValueError("The original probe must match the published frozen v1 SHA-256")
    original = json.loads(source)
    verify_model_source(DEFAULT_MODEL, DEFAULT_REVISION, args.model_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    v1_train = build_dataset(split="train")
    v2_train = build_temporal_dataset("train")
    validation = build_temporal_dataset("validation")
    train_rows = v1_train + v2_train
    rows = train_rows + validation
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Development row IDs must be unique across generations and splits")
    model = ResidualModel(DEFAULT_MODEL, DEFAULT_REVISION, 12, args.device, args.dtype, args.model_path)
    if model.hidden_size != original["hidden_size"] or original["layer"] != model.layer:
        raise ValueError("The frozen v1 probe is incompatible with the model")
    print(f"Extracting {len(train_rows)} train and {len(validation)} validation target states",
          flush=True)
    activations = model.encode_rows(rows, args.batch_size)
    x_train, x_val = np.split(activations, [len(train_rows)])
    probes, reports, scores = develop_probes(train_rows, x_train, validation, x_val, original)
    for name, probe in probes.items():
        filename = name + ".json"
        if name == "frozen-v1":
            (folder / filename).write_bytes(source)
        else:
            write_json(folder / filename, probe)
        reports[name].update({"file": filename, "sha256": digest(folder / filename)})
    write_jsonl(folder / "development.jsonl", rows)
    np.savez_compressed(folder / "activations.npz", row_ids=np.array(ids), activations=activations)
    pred_rows = []
    for i, row in enumerate(validation):
        pred = {k: row[k] for k in ("id", "group_id", "pair_id", "layout_id",
                                     "label", "alternative_location")}
        pred["scores"] = {family: float(values[i]) for family, values in scores.items()}
        pred_rows.append(pred)
    write_jsonl(folder / "validation-predictions.jsonl", pred_rows)
    selected = select_model(reports)
    selection = {
        "schema_version": "temporal-development.v1", "model_id": DEFAULT_MODEL,
        "model_revision": DEFAULT_REVISION, "layer": 12, "original_probe_sha256": BASELINE_SHA256,
        "models": reports, "selected": selected, "candidate_order": list(SELECTION_ORDER),
        "threshold_rule": THRESHOLD_RULE,
        "model_selection": ("maximize prior TP, absent TP, then minimize total FP, among "
                            "feasible candidates; exact ties resolve in candidate_order"),
        "loss": {"balanced-v2": "train-standardized logistic BCE, L2=0.1",
                 "prior-weighted-v2": "same loss; prior-location chart pairs replicated 3x before standardization"},
        "rows": {"v1_train": len(v1_train), "temporal_train": len(v2_train),
                 "temporal_validation": len(validation)},
        "row_order": ["v1_train", "temporal_train", "temporal_validation"],
        "validation_layouts": dict(sorted(Counter(r["layout_id"] for r in validation).items())),
        "device": args.device, "dtype": args.dtype, "threads": args.threads,
        "batch_size": args.batch_size, "seed": 1729, "attention": "eager",
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "python": platform.python_version(), "transformers": transformers.__version__,
        "torch": str(torch.__version__), "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "sha256": {path.name: digest(path) for path in sorted(folder.iterdir())},
    }
    write_json(folder / "selection.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--original-probe", type=Path, default=Path("artifacts/baseline/probe.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("thread count and batch size must be positive")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("use float32 for CPU runs")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.output.exists():
        parser.error("choose a new output directory for the atomic development run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temporal-dev-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        result = train(stage, args)
        stage.rename(args.output)
    print(f"Selected {result['selected']}; outputs in {args.output}", flush=True)


if __name__ == "__main__":
    main()
