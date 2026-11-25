"""Hand-computed fixtures for the independent evaluation. No real holdout is read."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.evaluate_temporal import (
    PROBES, audited_threshold, check_isolation, load_development, metrics_for,
    paired_transitions, predict, verified_activations,
)


def fixture_rows():
    rows = []
    for layout in ("A", "B"):
        for group in (0, 1):
            for order in ("current_first", "prior_first"):
                for label, location in ((0, "prior"), (1, "prior"), (0, "absent"), (1, "absent")):
                    i = len(rows)
                    rows.append({"id": f"r{i}", "group_id": f"{layout}-{group}",
                                 "pair_id": f"{layout}-{group}-{order}-{location}", "layout_id": layout,
                                 "field_id": "field", "chart_order": order, "split": "test",
                                 "target_text": "entry", "prompt": f"source {i}" + (" entry" if
                                 label == 0 or location == "prior" else ""),
                                 "label": label, "alternative_location": location})
    return rows


class TemporalAuditTests(unittest.TestCase):
    def test_threshold_requires_each_layout_and_uses_exact_tie_cutoff(self):
        # The tiny strata disallow even one FP. A pooled cap would permit one.
        rows, scores = [], []
        for layout in ("A", "B"):
            for i in range(9 if layout == "A" else 1):
                rows.append({"layout_id": layout, "label": 0, "alternative_location": "prior"})
                scores.append(0.4 if layout == "B" else 0.01)
            for loc, risk in (("prior", 0.9), ("prior", 0.4), ("absent", 0.7)):
                rows.append({"layout_id": layout, "label": 1, "alternative_location": loc})
                scores.append(risk)
        self.assertEqual(audited_threshold(rows, np.array(scores)), ((2, 2, 0), 0.7))

    def test_isolation_rejects_shared_content_and_group_ids(self):
        test = fixture_rows()
        dev = [{k: "dev-" + str(v) for k, v in test[0].items()}]
        check_isolation(test, dev)
        for key in ("id", "group_id", "prompt", "pair_id", "layout_id", "field_id"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                check_isolation(test, [{**dev[0], key: test[0][key]}])

    def test_digest_check_precedes_loading_source_or_selection(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw)
            (path / "selection.json").write_bytes(b"not even json")
            with self.assertRaisesRegex(ValueError, "selection SHA"):
                load_development(path, "0" * 64)

    def test_activations_reject_reordered_or_nonfinite_cache(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "cache.npz"
            rows = [{"id": "x"}, {"id": "y"}]
            np.savez(path, activations=[[0.], [1.]], row_ids=["y", "x"])
            with self.assertRaisesRegex(ValueError, "row IDs"):
                verified_activations(path, rows, 1)
            np.savez(path, activations=[[0.], [np.inf]], row_ids=["x", "y"])
            with self.assertRaisesRegex(ValueError, "invalid"):
                verified_activations(path, rows, 1)

    def test_pairwise_transitions_and_rate_denominators(self):
        rows = fixture_rows()
        probes = {name: {"weights": [1.0], "bias": 0.0,
                         "threshold": 0.6 if name == "frozen-v1" else 0.5} for name in PROBES}
        # The recalibration newly catches every prior substitution, and falsely flags
        # each supported first member of a prior pair. Both rules catch absent claims.
        x = np.array([[1.] if r["label"] == 1 and r["alternative_location"] == "absent"
                      else [0.] if r["alternative_location"] == "prior" else [-1.] for r in rows])
        pred = predict(rows, x, probes)
        result = metrics_for(pred, probes, "recalibrated-v1")
        paired = result["paired"]["frozen-v1"]["recalibrated-v1"]
        self.assertEqual(paired["prior_catch"]["newly_flagged"], 8)
        self.assertEqual(paired["prior_catch"]["group_bootstrap_95pct"], [1., 1.])
        self.assertEqual(paired["supported_false_alarm"]["total"], 16)
        self.assertEqual(paired["supported_false_alarm"]["newly_flagged"], 8)
        self.assertEqual(paired["absent_catch"]["both_flagged"], 8)
        self.assertEqual(result["overall"]["recalibrated-v1"]["false_positive_rate"], .5)
        self.assertEqual(result["by_chart_order"]["prior_first"]["lexical"]["absent_catch"]["flagged"], 4)
        self.assertEqual(len(result["paired_by_layout_and_order"]), 4)
        chosen = [r for r in pred if r["layout_id"] == "A" and r["chart_order"] == "current_first"]
        reverse = paired_transitions(chosen, "recalibrated-v1", "frozen-v1")
        self.assertEqual(reverse["prior_catch"]["newly_clear"], 2)
        self.assertEqual(reverse["prior_catch"]["rate_delta"], -1.)


if __name__ == "__main__":
    unittest.main()
