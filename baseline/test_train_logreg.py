"""Focused regression checks for data integrity and evaluation boundaries."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.train_logreg import (
    IntegrityError,
    evaluate,
    make_pipeline,
    read_split,
    select_best,
    sha256,
    validate_splits,
    verify_predictions,
    write_predictions,
)


class BaselineChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, name="train.csv", rows=None, columns=None):
        path = self.root / name
        if rows is None:
            rows = [("00001", "bad flight", "0"),
                    ("9007199254740993123", "NA", "1"),
                    ("3", "  good flight  ", "2")]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns or ["tweet_id", "text", "true_label"])
            writer.writerows(rows)
        return path

    def test_exact_ids_text_and_source_preservation(self):
        path = self.source()
        original_hash = sha256(path)
        split = read_split(path)
        self.assertEqual(split.ids, ["00001", "9007199254740993123", "3"])
        self.assertEqual(split.texts, ["bad flight", "NA", "  good flight  "])
        self.assertEqual(sha256(path), original_hash)

    def test_invalid_rows_fail_without_repair(self):
        for rows in (
            [("a", "tweet", "3")], [("a", "tweet", "1.0")],
            [("a", "   ", "0")], [("", "tweet", "0")],
            [("a", "tweet", "0"), ("a", "other", "1")],
            [("a", "tweet")], [("a", "tweet", "0", "extra")], [],
        ):
            with self.subTest(rows=rows):
                path = self.source(rows=rows)
                before = sha256(path)
                with self.assertRaises(IntegrityError):
                    read_split(path)
                self.assertEqual(sha256(path), before)

    def test_missing_and_duplicate_columns_fail(self):
        for columns in (["tweet_id", "text", "wrong"], ["tweet_id", "text", "true_label", "text"]):
            with self.subTest(columns=columns):
                with self.assertRaises(IntegrityError):
                    read_split(self.source(columns=columns))

    def test_cross_split_overlap_fails(self):
        train = read_split(self.source())
        val = read_split(self.source("val.csv", [("00001", "different text", "1")]))
        with self.assertRaisesRegex(IntegrityError, "overlapping tweet IDs"):
            validate_splits({"train": train, "val": val})

    def test_ties_use_smaller_c_then_unweighted(self):
        rows = [{"C": c, "class_weight": weight, "macro_f1": 0.5}
                for c in (10.0, 1.0, 0.1) for weight in ("balanced", None)]
        best = select_best(rows)
        self.assertEqual((best["C"], best["class_weight"]), (0.1, None))
        rows[0]["macro_f1"] = 0.6
        self.assertIs(select_best(rows), rows[0])

    def test_fixed_class_order_even_when_a_class_is_absent(self):
        metrics = evaluate(np.array([0, 0]), np.array([0, 0]))
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(metrics["per_class"]["2"]["f1"], 0)

    def test_held_out_tokens_do_not_enter_vocabulary(self):
        model = make_pipeline(1.0, None)
        model.fit(["bad delayed", "ordinary flight", "great flight"], [0, 1, 2])
        vectorizer = model.named_steps["tfidf"]
        original_vocabulary = dict(vectorizer.vocabulary_)
        original_idf = vectorizer.idf_.copy()
        model.predict(["validationonlytoken testonlytoken"])
        self.assertEqual(vectorizer.vocabulary_, original_vocabulary)
        np.testing.assert_array_equal(vectorizer.idf_, original_idf)
        self.assertNotIn("validationonlytoken", vectorizer.vocabulary_)
        self.assertNotIn("testonlytoken", vectorizer.vocabulary_)

    def test_export_verification_rejects_order_and_prediction_corruption(self):
        split = read_split(self.source())
        predictions = np.array([0, 2, 1])
        path = self.root / "predictions.csv"
        write_predictions(path, split, predictions)
        verify_predictions(path, split, predictions)
        original = path.read_text(encoding="utf-8")
        for bad in (
            original.replace("00001", "1"),
            original.replace("00001,0,0", "00001,0,3"),
            original.replace("00001,0,0", "00001,1,0"),
            "\n".join([original.splitlines()[0]] + list(reversed(original.splitlines()[1:]))) + "\n",
        ):
            with self.subTest(bad=bad):
                path.write_text(bad, encoding="utf-8")
                with self.assertRaises(IntegrityError):
                    verify_predictions(path, split, predictions)


if __name__ == "__main__":
    unittest.main()
