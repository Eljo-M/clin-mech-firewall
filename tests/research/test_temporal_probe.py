import unittest

import numpy as np

from mech_interp.data import build_dataset
from mech_interp.temporal_probe import (
    SELECTION_ORDER, select_model, threshold_for_validation, training_repetitions, validation_counts,
)


def example(label, layout, condition, split="validation"):
    return {"label": label, "layout_id": layout, "alternative_location": condition, "split": split}


class TemporalProbeTests(unittest.TestCase):
    def test_v1_split_filter_preserves_original_train_rows(self):
        train = build_dataset(split="train")
        self.assertEqual(len(train), 160)
        self.assertEqual(set(r["split"] for r in train), {"train"})
        self.assertEqual(len({r["id"] for r in train}), 160)

    def test_repeats_entire_prior_pairs_including_supported_controls(self):
        rows = [{**example(label, "layout", loc, "train"), "pair_id": loc}
                for loc in ("prior", "absent") for label in (0, 1)]
        np.testing.assert_array_equal(training_repetitions(rows, 3), [3, 3, 1, 1])
        with self.assertRaisesRegex(ValueError, "pair"):
            training_repetitions(rows[:-1], 3)
        with self.assertRaisesRegex(ValueError, "train"):
            training_repetitions([{**r, "split": "validation"} for r in rows], 1)

    def test_threshold_caps_each_layout_instead_of_pooled_false_alarms(self):
        rows = ([example(0, "a", "prior")] * 10 + [example(0, "b", "absent")] * 20
                + [example(1, "a", "prior")] * 2 + [example(1, "b", "absent")] * 2)
        risks = [0.90, 0.85] + [0.1] * 28 + [0.88, 0.8, 0.90, 0.95]
        found = threshold_for_validation(rows, risks)
        self.assertEqual(found, 0.88)
        approved = validation_counts(rows, risks, found)
        self.assertTrue(approved["feasible"])
        self.assertEqual(approved["by_location"]["prior"]["tp"], 1)
        self.assertEqual(approved["by_layout"]["a"]["fp"], 1)
        self.assertFalse(validation_counts(rows, risks, 0.8)["feasible"])

    def test_threshold_processes_identical_risks_as_one_block(self):
        rows = [example(1, "a", "prior"), example(0, "a", "prior"), example(1, "a", "absent")]
        self.assertEqual(threshold_for_validation(rows, [0.8, 0.8, 0.9]), 0.9)
        with self.assertRaisesRegex(ValueError, "No feasible"):
            threshold_for_validation(rows, [0.8, 1.0, 0.9])

    def test_selection_prioritizes_prior_recall_and_simplest_tie(self):
        def report(prior, absent, fp):
            return {"validation": {"feasible": True, "overall": {"fp": fp},
                    "by_location": {"prior": {"tp": prior}, "absent": {"tp": absent}}}}
        reports = {name: report(2, 8, 1) for name in SELECTION_ORDER}
        self.assertEqual(select_model(reports), "recalibrated-v1")
        reports["prior-weighted-v2"] = report(3, 1, 2)
        self.assertEqual(select_model(reports), "prior-weighted-v2")
        reports["balanced-v2"] = report(3, 1, 2)
        self.assertEqual(select_model(reports), "balanced-v2")


if __name__ == "__main__":
    unittest.main()
