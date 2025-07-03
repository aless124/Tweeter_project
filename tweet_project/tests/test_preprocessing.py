import csv
import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src import preprocessing as text_preprocessing
from src.preprocessing import (
    tokenize,
    build_clean_corpus,
    token_statistics,
    generate_wordcloud,
    clean_text,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tweets.csv')

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DATA_PATH, encoding='utf-8') as f:
            cls.rows = list(csv.DictReader(f))

    def test_basic_tokenize(self):
        tokens = tokenize("Hello!!! This is a test, testing  ???")
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
        try:
            import wordcloud
            result = generate_wordcloud(['hello', 'world'])
            self.assertIsNotNone(result)
        except ImportError:
            with self.assertRaises(ImportError):
                generate_wordcloud(['hello', 'world'])

    def test_clean_text_empty(self):
        self.assertEqual(clean_text(''), [])

    def test_tokens_length_and_stopwords(self):
        tokens = clean_text('This is just a simple TEST, with numbers 1234!')
        self.assertTrue(all(len(t) >= 3 for t in tokens))
        self.assertTrue(all(t not in text_preprocessing.STOPWORDS for t in tokens))

    def test_vocab_reduction_after_cleaning(self):
        import re
        raw_tokens = []
        cleaned_tokens = []
        pattern = re.compile(r"[A-Za-z]+")
        for r in self.rows:
            words = pattern.findall(r['text'].lower())
            raw_tokens.extend(words)
            cleaned_tokens.extend(clean_text(r['text']))
        self.assertLess(len(set(cleaned_tokens)), len(set(raw_tokens)))

    def test_stemming_effect(self):
        no_stem = []
        stem = []
        for r in self.rows:
            no_stem.extend(tokenize(r['text'], stem_words=False))
            stem.extend(tokenize(r['text'], stem_words=True))
        self.assertLessEqual(len(set(stem)), len(set(no_stem)))

if __name__ == '__main__':
    unittest.main()
