import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from classification_pipeline import (
    train_evaluate,
    cross_validate,
    train_predict,
    SKLEARN_AVAILABLE,
)

DATA_PATH = 'train (1).csv'


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
class TestClassification(unittest.TestCase):
    def test_train_evaluate_metrics(self):
        metrics = train_evaluate(DATA_PATH)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

    def test_cross_validate_accuracy_range(self):
        score = cross_validate(DATA_PATH, vectorizer='count', model='svm', cv=3)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_prediction_shape(self):
        preds, truth = train_predict(DATA_PATH, test_size=0.25)
        self.assertEqual(len(preds), len(truth))
        self.assertGreater(len(preds), 0)

    def test_handles_empty_or_short_text(self):
        import tempfile, csv
        with tempfile.NamedTemporaryFile('w+', newline='', delete=False) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
            writer.writerow([1, '', '', '', 0])
            writer.writerow([2, '', '', 'ok', 1])
            writer.writerow([3, '', '', 'another normal tweet', 0])
            path = tmp.name
        metrics = train_evaluate(path)
        self.assertIn('accuracy', metrics)

if __name__ == '__main__':
    unittest.main()
