"""Preregistered temporal probe development rules; the deployment scorer stays token-local."""

from collections import Counter

import numpy as np

from .probe import classification_metrics, fit_probe, score


SELECTION_ORDER = ("recalibrated-v1", "balanced-v2", "prior-weighted-v2")
BASELINE_SHA256 = "3c7bd6c3ab4547618a013f4aae07e1ad69708ca151064ba3b8da8b230d945350"
THRESHOLD_RULE = ("on validation, allow at most 10% supported false alarms in each layout; "
                  "maximize prior true positives, then absent true positives, then minimize "
                  "total false positives, then choose the largest tied threshold")


def require_split(rows, split):
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError(f"Expected only {split} rows")
    for row in rows:
        if row["label"] not in (0, 1) or row["alternative_location"] not in ("prior", "absent"):
            raise ValueError("Unexpected label or temporal condition")


def training_repetitions(rows, prior_weight):
    """Replicate intact supported/unsupported pairs from charts with a prior-visit distractor."""
    require_split(rows, "train")
    if prior_weight not in (1, 3):
        raise ValueError("Training uses prior pair multiplicity 1 or 3")
    counts = Counter((r["pair_id"], r["alternative_location"], r["label"]) for r in rows)
    pair_sizes = Counter(r["pair_id"] for r in rows)
    if (any(n != 2 for n in pair_sizes.values())
            or any(n != 1 or counts[pair, location, 1 - label] != 1
                   for (pair, location, label), n in counts.items())):
        raise ValueError("Every chart pair must contain both labels in a single mention condition")
    return np.array([prior_weight if r["alternative_location"] == "prior" else 1 for r in rows])


def validation_counts(rows, scores, threshold):
    """Raw counts used in constrained operating-point and between-model selection."""
    require_split(rows, "validation")
    y = np.array([r["label"] for r in rows])
    risks = np.asarray(scores, dtype=np.float64)
    stats = classification_metrics(y, risks, threshold)
    predicted = risks >= threshold
    locations = np.array([r["alternative_location"] for r in rows])
    layouts = np.array([r["layout_id"] for r in rows])
    by_layout = {}
    for layout in sorted(set(layouts)):
        supported = (layouts == layout) & (y == 0)
        n, fp = int(supported.sum()), int((supported & predicted).sum())
        if n == 0:
            raise ValueError(f"Missing supported controls in validation layout {layout}")
        by_layout[str(layout)] = {"n_supported": n, "fp": fp, "fpr": fp / n}
    by_location = {}
    for condition in ("prior", "absent"):
        mask = (locations == condition) & (y == 1)
        n, tp = int(mask.sum()), int((mask & predicted).sum())
        if n == 0:
            raise ValueError(f"Validation is missing {condition} errors")
        by_location[condition] = {"n_errors": n, "tp": tp, "recall": tp / n}
    return {"overall": stats, "by_layout": by_layout, "by_location": by_location,
            "feasible": all(10 * x["fp"] <= x["n_supported"] for x in by_layout.values())}


def threshold_for_validation(rows, scores):
    """Sweep tied risks together. Integer arithmetic enforces each layout's exact FPR cap."""
    require_split(rows, "validation")
    risks = np.asarray(scores, dtype=np.float64)
    if (risks.shape != (len(rows),) or not np.isfinite(risks).all()
            or np.any((risks < 0) | (risks > 1))):
        raise ValueError("Expected one finite risk in [0, 1] per validation example")
    by_layout = Counter(r["layout_id"] for r in rows if r["label"] == 0)
    fp = Counter()
    tp = Counter()
    fp_total = 0
    best = None
    thresholds = np.unique(np.append(risks, 1.0))[::-1]
    order = np.argsort(-risks, kind="stable")
    index = 0
    for threshold in thresholds:
        while index < len(order) and risks[order[index]] >= threshold:
            row = rows[order[index]]
            if row["label"] == 0:
                fp[row["layout_id"]] += 1
                fp_total += 1
            else:
                tp[row["alternative_location"]] += 1
            index += 1
        if any(10 * fp[layout] > n for layout, n in by_layout.items()):
            break  # Lowering an inclusive threshold can only add false positives.
        key = (tp["prior"], tp["absent"], -fp_total, float(threshold))
        if best is None or key > best:
            best = key
    if best is None:
        raise ValueError("No feasible threshold in [0, 1] for all validation layouts")
    return best[-1]


def comparison_key(model_validation):
    metrics = model_validation["by_location"]
    return (metrics["prior"]["tp"], metrics["absent"]["tp"],
            -model_validation["overall"]["fp"])


def select_model(reports):
    for name in SELECTION_ORDER:
        if not reports[name]["validation"]["feasible"]:
            raise ValueError(f"Constrained candidate {name} is infeasible")
    return max(SELECTION_ORDER, key=lambda name: comparison_key(reports[name]["validation"]))


def develop_probes(train_rows, train_x, val_rows, val_x, original):
    require_split(train_rows, "train")
    require_split(val_rows, "validation")
    if {r["group_id"] for r in train_rows} & {r["group_id"] for r in val_rows}:
        raise ValueError("The same chart group appears in training and validation")
    if {r["prompt"] for r in train_rows} & {r["prompt"] for r in val_rows}:
        raise ValueError("The same chart prompt appears in training and validation")
    if len(train_rows) != len(train_x) or len(val_rows) != len(val_x):
        raise ValueError("Expected one activation vector per row")
    probes = {"frozen-v1": original, "recalibrated-v1": dict(original)}
    y = np.array([r["label"] for r in train_rows])
    for family, weight in (("balanced-v2", 1), ("prior-weighted-v2", 3)):
        repeats = training_repetitions(train_rows, weight)
        # Explicit replication gives the same weight to each pair member throughout
        # standardization, the binary cross-entropy term and the intercept fit.
        indices = np.repeat(np.arange(len(train_rows)), repeats)
        trained = fit_probe(train_x[indices], y[indices], l2=0.1)
        probes[family] = {**original, **trained, "training_rows": len(train_rows),
                          "weighted_training_rows": len(indices), "prior_pair_multiplicity": weight}
    reports, scores = {}, {}
    for family, probe in probes.items():
        risks = score(val_x, probe)
        if family != "frozen-v1":
            probe["threshold"] = threshold_for_validation(val_rows, risks)
            probe["threshold_rule"] = THRESHOLD_RULE
            probe["development_family"] = family
        reports[family] = {"validation": validation_counts(val_rows, risks, probe["threshold"]),
                           "threshold": probe["threshold"]}
        scores[family] = risks
    return probes, reports, scores
