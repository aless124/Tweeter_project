import csv
import unittest

# ensure module path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from text_preprocessing import tokenize, build_clean_corpus, token_statistics, generate_wordcloud

DATA_PATH = 'train (1).csv'

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DATA_PATH, encoding='utf-8') as f:
            cls.rows = list(csv.DictReader(f))

    def test_basic_tokenize(self):
        tokens = tokenize("Hello!!! This is a test, testing 123 ???")
        self.assertEqual(tokens, ['hello', 'test', 'test'])

    def test_corpus_stats(self):
        tokens = []
        for r in self.rows:
            tokens.extend(tokenize(r['text']))
        stats = token_statistics(tokens)
        self.assertEqual(stats['total'], 84482)
        self.assertEqual(stats['unique'], 19329)
        self.assertEqual(stats['once'], 13665)

    def test_clean_corpus_length(self):
        cleaned = build_clean_corpus([r['text'] for r in self.rows])
        self.assertEqual(len(cleaned), len(self.rows))

    def test_wordcloud_import(self):
        with self.assertRaises(ImportError):
            generate_wordcloud(['hello', 'world'])

if __name__ == '__main__':
    unittest.main()
