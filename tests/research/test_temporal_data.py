"""Check identities, matched controls, offsets, and independence of temporal splits."""

import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from mech_interp.data import build_dataset
from mech_interp.temporal_data import DATA_DIR, SOURCE_HASHES, build_temporal_dataset


class TemporalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = {split: build_temporal_dataset(split) for split in SOURCE_HASHES}

    def test_split_identity_and_disjointness(self):
        original = build_dataset()
        previous = {
            field: {r[field].casefold() for r in original}
            for field in ("id", "pair_id", "group_id", "prompt", "layout_id", "field_id", "target_text")
        }
        sizes = {"train": (768, 4, 6), "validation": (384, 3, 4), "test": (640, 4, 5)}
        for split, rows in self.splits.items():
            with self.subTest(split=split):
                self.assertEqual(sizes[split], (len(rows), len({r["layout_id"] for r in rows}),
                                                len({r["field_id"] for r in rows})))
                self.assertEqual({r["split"] for r in rows}, {split})
                self.assertEqual({r["source_sha256"] for r in rows}, {SOURCE_HASHES[split]})
                self.assertEqual(len({r["id"] for r in rows}), len(rows))
                self.assertEqual(len({r["prompt"] for r in rows}) * 2, len(rows))
                for key, used in previous.items():
                    values = {r[key].casefold() for r in rows}
                    self.assertTrue(values.isdisjoint(used), f"shared {key} in {split}")
                    used.update(values)

    def test_target_and_evidence_locations(self):
        for split, rows in self.splits.items():
            with self.subTest(split=split):
                for row in rows:
                    text = row["prompt"]
                    self.assertTrue(text.startswith(row["chart"]))
                    for role in ("current", "prior"):
                        start, end = row[role + "_start"], row[role + "_end"]
                        self.assertTrue(0 <= start < end <= len(row["chart"]))
                        self.assertTrue(text[start:end] == row[role + "_value"])
                    self.assertEqual(row["current_start"] < row["prior_start"],
                                     row["chart_order"] == "current_first")
                    self.assertTrue(row["current_value"] != row["prior_value"])
                    self.assertTrue(row["target_text"] == row["completion"][row["target_start"]:row["target_end"]])
                    self.assertTrue(row["candidate_value"] == row["completion"].strip().removesuffix("."))
                    self.assertEqual(int(row["candidate_value"] != row["current_value"]), row["label"])
                    appears = row["target_text"].casefold() in text.casefold()
                    self.assertEqual(appears, row["label"] == 0 or row["alternative_location"] == "prior")
                    if row["label"] == 1:
                        self.assertEqual(row["candidate_value"] == row["prior_value"],
                                         row["alternative_location"] == "prior")

    def test_counterbalancing_and_group_integrity(self):
        for split, rows in self.splits.items():
            groups, pairs = defaultdict(list), defaultdict(list)
            for row in rows:
                groups[row["group_id"]].append(row)
                pairs[row["pair_id"]].append(row)
            with self.subTest(split=split):
                for members in pairs.values():
                    self.assertEqual(len(members), 2)
                    self.assertEqual({r["label"] for r in members}, {0, 1})
                    self.assertEqual(len({r["prompt"] for r in members}), 1)
                    self.assertEqual(len({r["completion"] for r in members}), 2)
                for members in groups.values():
                    self.assertEqual(len(members), 16)
                    self.assertEqual(len({(r["layout_id"], r["field_id"]) for r in members}), 1)
                    self.assertEqual(len({r["pair_id"] for r in members}), 8)
                    for order in ("current_first", "prior_first"):
                        for location in ("prior", "absent"):
                            quartet = [r for r in members if r["chart_order"] == order and
                                       r["alternative_location"] == location]
                            self.assertEqual(len(quartet), 4)
                            supported = Counter(r["completion"] for r in quartet if r["label"] == 0)
                            substituted = Counter(r["completion"] for r in quartet if r["label"] == 1)
                            self.assertTrue(supported == substituted)

    def test_explicit_selector_only_opens_requested_file_and_detects_edits(self):
        for split in SOURCE_HASHES:
            with self.subTest(split=split), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                name = f"temporal_{split}_v1.json"
                source = directory / name
                source.write_bytes((DATA_DIR / name).read_bytes())
                with patch("mech_interp.temporal_data.DATA_DIR", directory):
                    rows = build_temporal_dataset(split)
                    self.assertEqual(len(rows), len(self.splits[split]))
                    content = json.loads(source.read_bytes())
                    source.write_text(json.dumps(content, indent=4), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "locked"):
                        build_temporal_dataset(split)
        with self.assertRaisesRegex(ValueError, "split"):
            build_temporal_dataset("all")


if __name__ == "__main__":
    unittest.main()
