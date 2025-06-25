import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.modeling import (
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tweets.csv')
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


if __name__ == '__main__':
    unittest.main()
