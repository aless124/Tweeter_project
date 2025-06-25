import os
import sys
import unittest

# ensure the root directory is on the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset_analysis import (
    load_dataset,
    dataset_shape,
    missing_values,
    duplicate_count,
    text_length_stats,
    target_distribution,
    all_texts_non_empty,
    unique_targets,
)

DATA_PATH = 'train (1).csv'

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.columns = load_dataset(DATA_PATH)

    def test_columns_present(self):
        self.assertIn('text', self.columns)
        self.assertIn('target', self.columns)

    def test_shape(self):
        rows, cols = dataset_shape(self.rows, self.columns)
        self.assertEqual(rows, 7613)
        self.assertEqual(cols, 5)

    def test_missing_values(self):
        missing = missing_values(self.rows, self.columns)
        self.assertEqual(missing['text'], 0)
        self.assertEqual(missing['target'], 0)

    def test_duplicate_ids(self):
        self.assertEqual(duplicate_count(self.rows), 0)

    def test_texts_non_empty(self):
        self.assertTrue(all_texts_non_empty(self.rows))

    def test_target_classes(self):
        classes = unique_targets(self.rows)
        self.assertEqual(sorted(classes), ['0', '1'])

    def test_length_stats(self):
        min_len, max_len, mean_len = text_length_stats(self.rows)
        self.assertGreaterEqual(min_len, 1)
        self.assertLessEqual(max_len, 160)
        self.assertGreater(mean_len, 5)

if __name__ == '__main__':
    unittest.main()
