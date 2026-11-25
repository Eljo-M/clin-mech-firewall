"""Evaluate a SHA-pinned development decision on fresh temporal grounding layouts."""

import argparse
import hashlib
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.validate_trace import parse_json, validate_probe
from mech_interp.probe import classification_metrics, score


PROBES = ("frozen-v1", "recalibrated-v1", "balanced-v2", "prior-weighted-v2")
METHODS = (*PROBES, "lexical")
V1_SHA256 = "3c7bd6c3ab4547618a013f4aae07e1ad69708ca151064ba3b8da8b230d945350"
TEST_SOURCE_SHA256 = "176d849ee2504d425d604bba6037d433be2ad0e5295aa3eda75c037d6ecaaa20"
BOOTSTRAP_SEED = 1729
BOOTSTRAP_DRAWS = 2000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_from(path):
    return [parse_json(line) for line in path.read_bytes().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_activations(path, rows, width):
    with np.load(path, allow_pickle=False) as cache:
        require(cache["row_ids"].tolist() == [row["id"] for row in rows], "cache row IDs differ")
        matrix = cache["activations"]
    require(matrix.shape == (len(rows), width) and np.isfinite(matrix).all(), "invalid activation cache")
    return matrix


def audited_threshold(rows, scores):
    """Brute-force validation selection, independent of the fitting implementation."""
    y = np.array([r["label"] for r in rows])
    prior = (y == 1) & np.array([r["alternative_location"] == "prior" for r in rows])
    absent = (y == 1) & ~prior
    layouts = np.array([r["layout_id"] for r in rows])
    support = [(y == 0) & (layouts == layout) for layout in set(layouts)]
    require(all(mask.any() for mask in support), "validation has an empty supported stratum")
    best = None
    for threshold in np.unique(np.append(scores, 1.0)):
        flagged = scores >= threshold
        if any(10 * np.count_nonzero(flagged & mask) > int(mask.sum()) for mask in support):
            continue
        outcome = (int((flagged & prior).sum()), int((flagged & absent).sum()),
                   -int((flagged & (y == 0)).sum()))
        best = max(best, (outcome, float(threshold))) if best is not None else (outcome, float(threshold))
    require(best is not None, "no eligible validation threshold")
    return best


def load_development(folder, freeze_sha):
    selection_path = folder / "selection.json"
    require(sha(selection_path) == freeze_sha, "selection SHA differs from the frozen decision")
    selection = parse_json(selection_path.read_bytes())
    for name in ("development.jsonl", "validation-predictions.jsonl", "activations.npz"):
        require(sha(folder / name) == selection["sha256"][name], f"frozen development {name} SHA mismatch")
    models = selection["models"]
    require(set(models) == set(PROBES), "unexpected set of development models")
    probes = {}
    for name in PROBES:
        spec = models[name]
        require(spec["file"] == name + ".json", "unexpected model filename")
        require(spec["sha256"] == selection["sha256"][spec["file"]], "inconsistent probe digests in selection")
        path = folder / spec["file"]
        probe = parse_json(path.read_bytes())
        validate_probe({**probe, "probe_id": "sha256:" + spec["sha256"]}, path)
        probes[name] = probe
    require(models["frozen-v1"]["sha256"] == V1_SHA256, "published v1 probe has changed")
    first = probes["frozen-v1"]
    for probe in probes.values():
        require(all(probe[k] == first[k] for k in ("model_id", "model_revision", "layer", "hidden_size")),
                "probes require different residuals")
    require(all(first[k] == probes["recalibrated-v1"][k] for k in ("weights", "bias")),
            "recalibrated v1 weights or intercept changed")
    development = rows_from(folder / "development.jsonl")
    ids = [r["id"] for r in development]
    require(len(ids) == len(set(ids)) and {r["split"] for r in development} == {"train", "validation"},
            "development contains duplicate IDs or a forbidden split")
    for key in ("group_id", "pair_id", "layout_id", "prompt"):
        train_keys = {r[key] for r in development if r["split"] == "train"}
        require(train_keys.isdisjoint(r[key] for r in development if r["split"] == "validation"),
                f"development train and validation overlap: {key}")
    matrix = verified_activations(folder / "activations.npz", development, first["hidden_size"])
    validation = [r for r in development if r["split"] == "validation"]
    val_index = [i for i, row in enumerate(development) if row["split"] == "validation"]
    logged = rows_from(folder / "validation-predictions.jsonl")
    require([r["id"] for r in logged] == [r["id"] for r in validation], "validation row IDs differ")
    outcomes = {}
    for name, probe in probes.items():
        stored = np.array([r["scores"][name] for r in logged])
        require(np.allclose(stored, score(matrix[val_index], probe), rtol=0, atol=1e-10),
                f"{name} validation scores do not reproduce")
        if name != "frozen-v1":
            quality, threshold = audited_threshold(validation, stored)
            require(threshold == probe["threshold"], f"{name} validation threshold does not reproduce")
            outcomes[name] = quality
    choice = max(PROBES[1:], key=outcomes.__getitem__)
    require(choice == selection["selected"], "primary model differs from the validation decision")
    return selection, probes, development


def check_isolation(test, dev):
    require(bool(test) and all(r["split"] == "test" for r in test), "only held-out rows may be evaluated")
    require(len({r["id"] for r in test}) == len(test), "duplicate held-out ID")
    for key in ("id", "prompt", "group_id", "pair_id", "layout_id", "field_id"):
        require({r[key] for r in test}.isdisjoint(r[key] for r in dev), f"test overlaps development: {key}")
    require(all(r["label"] in (0, 1) and r["alternative_location"] in ("prior", "absent")
                and r["chart_order"] in ("current_first", "prior_first") for r in test), "unknown test stratum")


def predict(rows, matrix, probes):
    probabilities = {}
    for name, probe in probes.items():
        scores = score(matrix, probe)
        # Scalar summation checks the deployed raw-coordinate formula independently of BLAS.
        for vector, risk in zip(matrix, scores, strict=True):
            logit = math.fsum(float(x) * w for x, w in zip(vector, probe["weights"], strict=True)) + probe["bias"]
            require(abs(float(risk) - (0.5 + 0.5 * math.tanh(logit / 2))) < 1e-10,
                    f"{name} independent raw-score check failed")
        probabilities[name] = scores
    predictions = []
    for i, r in enumerate(rows):
        scores = {name: float(values[i]) for name, values in probabilities.items()}
        scores["lexical"] = float(r["target_text"].casefold() not in r["prompt"].casefold())
        flags = {name: int(value >= (0.5 if name == "lexical" else probes[name]["threshold"]))
                 for name, value in scores.items()}
        predictions.append({**{k: r[k] for k in ("id", "group_id", "pair_id", "layout_id", "field_id",
                                                 "chart_order", "label", "alternative_location")},
                            "scores": scores, "flags": flags})
    return predictions


def condition_masks(rows):
    y = np.array([r["label"] for r in rows])
    prior = np.array([r["alternative_location"] == "prior" for r in rows])
    return {"prior_catch": (y == 1) & prior, "absent_catch": (y == 1) & ~prior,
            "supported_false_alarm": y == 0}


def summarize(rows, thresholds):
    labels = [r["label"] for r in rows]
    masks = condition_masks(rows)
    result = {}
    for method in thresholds:
        risk = [r["scores"][method] for r in rows]
        flags = np.array([r["flags"][method] for r in rows], dtype=bool)
        rates = {}
        for name, mask in masks.items():
            n, hits = int(mask.sum()), int((flags & mask).sum())
            rates[name] = {"flagged": hits, "total": n, "rate": hits / n if n else None}
        result[method] = {**classification_metrics(labels, risk, thresholds[method]), **rates}
    return result


def transition_counts(old, new, mask):
    total = int(mask.sum())
    gains = int((mask & ~old & new).sum())
    losses = int((mask & old & ~new).sum())
    return {"total": total, "both_flagged": int((mask & old & new).sum()),
            "both_clear": int((mask & ~old & ~new).sum()), "newly_flagged": gains,
            "newly_clear": losses, "rate_delta": (gains - losses) / total if total else None}


def paired_transitions(rows, reference, candidate, with_intervals=True):
    """Count exact same-row transitions and resample matched groups within each layout."""
    old = np.array([r["flags"][reference] for r in rows], dtype=bool)
    new = np.array([r["flags"][candidate] for r in rows], dtype=bool)
    masks = condition_masks(rows)
    result = {name: transition_counts(old, new, mask) for name, mask in masks.items()}
    if not with_intervals:
        return result
    group_layout = {}
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        group, layout = row["group_id"], row["layout_id"]
        require(group_layout.setdefault(group, layout) == layout, "group crosses held-out layouts")
        groups[group].append(i)
    group_keys = sorted(groups)
    # Each bootstrap draw retains the observed layout mix. Interpret conditional on these layouts.
    by_layout = defaultdict(list)
    for i, group in enumerate(group_keys):
        by_layout[group_layout[group]].append(i)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.hstack([rng.choice(indices, (BOOTSTRAP_DRAWS, len(indices)), replace=True)
                       for indices in by_layout.values()])
    for name, mask in masks.items():
        gains, losses = mask & ~old & new, mask & old & ~new
        numerator = np.array([int(gains[idx].sum()) - int(losses[idx].sum()) for idx in
                              (groups[g] for g in group_keys)])
        denominator = np.array([int(mask[groups[g]].sum()) for g in group_keys])
        denom_draws = denominator[draws].sum(axis=1)
        valid = denom_draws > 0
        deltas = numerator[draws].sum(axis=1)[valid] / denom_draws[valid]
        result[name]["group_bootstrap_95pct"] = np.quantile(deltas, [0.025, 0.975]).tolist() if len(deltas) else None
    return result


def metrics_for(predictions, probes, selected):
    thresholds = {name: p["threshold"] for name, p in probes.items()} | {"lexical": 0.5}
    by_layout, by_order, joint = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in predictions:
        layout, order = row["layout_id"], row["chart_order"]
        by_layout[layout].append(row)
        by_order[order].append(row)
        joint[layout, order].append(row)
    return {
        "schema_version": "temporal-test.v1", "primary": selected, "thresholds": thresholds,
        "rows": len(predictions), "groups": len({r["group_id"] for r in predictions}),
        "overall": summarize(predictions, thresholds),
        "by_layout": {k: summarize(v, thresholds) for k, v in sorted(by_layout.items())},
        "by_chart_order": {k: summarize(v, thresholds) for k, v in sorted(by_order.items())},
        "by_layout_and_order": [{"layout_id": k[0], "chart_order": k[1], "methods": summarize(v, thresholds)}
                                for k, v in sorted(joint.items())],
        "paired": {reference: {method: paired_transitions(predictions, reference, method)
                               for method in METHODS if method != reference}
                   for reference in PROBES[:2]},
        "paired_by_layout_and_order": [
            {"layout_id": layout, "chart_order": order, "reference": "frozen-v1",
             "methods": {m: paired_transitions(v, "frozen-v1", m, False) for m in METHODS[1:]}}
            for (layout, order), v in sorted(joint.items())
        ],
        "bootstrap": {"replicates": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                      "unit": "complete matched group; resampled within each held-out layout",
                      "interval": "percentile; conditional on the published layouts and frozen thresholds"},
    }


def run(folder, development_dir, selection, probes, dev_rows, model_path):
    import torch
    import transformers

    from eval.run import write_json, write_jsonl
    from mech_interp.model import ResidualModel
    from mech_interp.snapshot import verify_model_source
    from mech_interp.temporal_data import build_temporal_dataset

    reference = probes["frozen-v1"]
    model_id, revision, layer = (reference[k] for k in ("model_id", "model_revision", "layer"))
    verify_model_source(model_id, revision, model_path)
    device, dtype = selection["device"], selection["dtype"]
    if device == "cuda":
        require(torch.cuda.is_available(), "the frozen selection used CUDA; no CUDA device is available")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(selection.get("threads", 4))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = build_temporal_dataset("test")
    check_isolation(rows, dev_rows)
    source = {r["source_sha256"] for r in rows}
    require(source == {TEST_SOURCE_SHA256}, "test source differs from the pre-registered file")
    model = ResidualModel(model_id, revision, layer, device, dtype, model_path)
    matrix = model.encode_rows(rows, selection["batch_size"])
    np.savez_compressed(folder / "activations.npz", row_ids=np.array([r["id"] for r in rows]),
                        activations=matrix)
    write_jsonl(folder / "dataset.jsonl", rows)
    predictions = predict(rows, matrix, probes)
    write_jsonl(folder / "predictions.jsonl", predictions)
    metrics = metrics_for(predictions, probes, selection["selected"])
    write_json(folder / "metrics.json", metrics)
    manifest = {
        "schema_version": "temporal-test-run.v1", "selection_sha256": sha(development_dir / "selection.json"),
        "model_id": model_id, "model_revision": revision, "layer": layer, "device": device, "dtype": dtype,
        "batch_size": selection["batch_size"], "source_sha256": source.pop(),
        "versions": {"torch": str(torch.__version__), "transformers": transformers.__version__, "numpy": np.__version__},
        "development_sha256": {name: sha(development_dir / name) for name in
                               ("development.jsonl", "validation-predictions.jsonl", "activations.npz")},
        "probe_sha256": {name: info["sha256"] for name, info in selection["models"].items()},
        "sha256": {p.name: sha(p) for p in sorted(folder.iterdir())},
    }
    write_json(folder / "manifest.json", manifest)


def audit(folder, development_dir, selection_sha, selection, probes, dev_rows):
    from mech_interp.temporal_data import build_temporal_dataset

    manifest = parse_json((folder / "manifest.json").read_bytes())
    require(manifest["selection_sha256"] == selection_sha, "test archive names a different frozen decision")
    for name in ("activations.npz", "dataset.jsonl", "predictions.jsonl", "metrics.json"):
        require(sha(folder / name) == manifest["sha256"][name], f"{name} SHA mismatch")
    for name in ("development.jsonl", "validation-predictions.jsonl", "activations.npz"):
        require(sha(development_dir / name) == manifest["development_sha256"][name], f"development {name} SHA mismatch")
    require(manifest["probe_sha256"] == {n: s["sha256"] for n, s in selection["models"].items()},
            "archive probe digests differ from selection")
    rows = rows_from(folder / "dataset.jsonl")
    check_isolation(rows, dev_rows)
    require({r["source_sha256"] for r in rows} == {manifest["source_sha256"]} == {TEST_SOURCE_SHA256},
            "test source digest differs")
    require(rows == build_temporal_dataset("test"), "archived rows differ from the published source")
    matrix = verified_activations(folder / "activations.npz", rows, probes["frozen-v1"]["hidden_size"])
    regenerated = predict(rows, matrix, probes)
    stored = rows_from(folder / "predictions.jsonl")
    require(len(stored) == len(regenerated), "wrong number of saved predictions")
    for old, new in zip(stored, regenerated, strict=True):
        require(set(old) == set(new) and all(old[k] == new[k] for k in new if k != "scores"),
                "predictions or row IDs do not reproduce")
        require(set(old["scores"]) == set(METHODS) and
                all(abs(old["scores"][k] - new["scores"][k]) < 1e-10 for k in METHODS),
                "raw predictor scores do not reproduce")
    require(metrics_for(stored, probes, selection["selected"]) == parse_json((folder / "metrics.json").read_bytes()),
            "reported paired counts or metrics do not reproduce")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path, help="frozen development directory")
    parser.add_argument("--selection-sha256", required=True, help="published SHA-256 of selection.json")
    parser.add_argument("--output", required=True, type=Path, help="new test directory, or existing with --audit-only")
    parser.add_argument("--model-path", type=Path, help="verified local snapshot of the pinned Hub revision")
    parser.add_argument("--audit-only", action="store_true", help="recompute saved scores and results without the LM")
    args = parser.parse_args()
    selection, probes, dev = load_development(args.development, args.selection_sha256)
    if args.audit_only:
        audit(args.output, args.development, args.selection_sha256, selection, probes, dev)
    else:
        if args.output.exists():
            parser.error("choose a new --output directory or use --audit-only")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".temporal-evaluation-", dir=args.output.parent) as temp:
            stage = Path(temp) / "test"
            stage.mkdir()
            run(stage, args.development, selection, probes, dev, args.model_path)
            audit(stage, args.development, args.selection_sha256, selection, probes, dev)
            stage.rename(args.output)
    print(f"Verified {args.output}; frozen primary: {selection['selected']}")


if __name__ == "__main__":
    main()
