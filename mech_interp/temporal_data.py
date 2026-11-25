"""Build exactly one split of the independently authored temporal grounding benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

from .data import Example


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SOURCE_HASHES = {
    "train": "4f1be5940f63fde4280456f21aa5825eb374a496750432852f9c5dc7d2d9d78f",
    "validation": "730328ccacd7ccd163200564c9c4745dbad3fe829a67e239dc96c995f9fca9c1",
    "test": "176d849ee2504d425d604bba6037d433be2ad0e5295aa3eda75c037d6ecaaa20",
}
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
ORDERS = {"current_first": ("current", "prior"), "prior_first": ("prior", "current")}


class TemporalExample(Example):
    chart_order: str
    current_start: int
    current_end: int
    prior_start: int
    prior_end: int
    source_sha256: str


def _chart(layout, field, order, current, prior):
    text = layout["header"].format(field=field["label"])
    spans = {}
    for position, role in enumerate(ORDERS[order]):
        if position and layout["bridge"]:
            text += "\n" + layout["bridge"]
        prefix, suffix = layout[role].split("{value}")
        text += "\n" + prefix
        value = current if role == "current" else prior
        start = len(text)
        text += value
        spans[role] = (start, len(text))
        text += suffix
    return text, spans


def _examples_for_group(source, layout, field, left, right):
    split = source["split"]
    values = field["values"]
    group = f"temporal-v1.{split}.{layout['id']}.{field['id']}.{left}-{right}"
    spare = [i for i in range(len(values)) if i not in (left, right)]
    question = field["question"]
    ask = layout["ask"].format(question=question, question_lower=question[0].lower() + question[1:])
    for condition in ("prior", "absent"):
        for order in ORDERS:
            for position, (current, alternate) in enumerate(((left, right), (right, left))):
                prior = alternate if condition == "prior" else spare[position]
                chart, spans = _chart(layout, field, order, values[current], values[prior])
                pair = f"{group}.{condition}.{order}.current-{current}"
                for label, candidate in enumerate((current, alternate)):
                    completion = " " + values[candidate] + "."
                    target = field["targets"][candidate]
                    start = completion.index(target)
                    yield TemporalExample(
                        id=f"{pair}.candidate-{candidate}", pair_id=pair, group_id=group, split=split,
                        prompt=chart + "\n\n" + ask, completion=completion, label=label,
                        target_text=target, target_start=start, target_end=start + len(target),
                        current_value=values[current], prior_value=values[prior],
                        candidate_value=values[candidate], alternative_location=condition,
                        chart=chart, chart_order=order, layout_id=layout["id"], field_id=field["id"],
                        current_start=spans["current"][0], current_end=spans["current"][1],
                        prior_start=spans["prior"][0], prior_end=spans["prior"][1],
                        source_version=source["version"], source_sha256=SOURCE_HASHES[split],
                    )


def build_temporal_dataset(split: str) -> list[TemporalExample]:
    """Read only the named immutable split. Every pair and group stays within that split."""
    if split not in SOURCE_HASHES:
        raise ValueError("split must be one of train, validation, or test")
    path = DATA_DIR / f"temporal_{split}_v1.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[split]:
        raise ValueError(f"{split} source differs from the locked temporal benchmark manifest")
    source = json.loads(raw)
    if source["split"] != split or source["version"] != "temporal-chart-v1":
        raise ValueError(f"Incorrect identity for the {split} source manifest")
    rows = []
    for layout in source["layouts"]:
        for field in source["fields"]:
            for left, right in MATCHINGS[layout["matching"]]:
                rows.extend(_examples_for_group(source, layout, field, left, right))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SOURCE_HASHES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = {DATA_DIR / f"temporal_{split}_v1.json" for split in SOURCE_HASHES}
    if args.output.resolve() in manifests:
        parser.error("output path would overwrite a locked source manifest")
    rows = build_temporal_dataset(args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} {args.split} rows to {args.output}")


if __name__ == "__main__":
    main()
