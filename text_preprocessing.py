import re
from collections import Counter
from typing import Iterable, List, Dict

# Simple stopword list (subset of common English stopwords)
STOPWORDS = {
    'the', 'and', 'is', 'in', 'to', 'a', 'of', 'for', 'on', 'with', 'at', 'by',
    'an', 'be', 'this', 'that', 'it', 'from', 'as', 'are', 'was', 'were', 'or',
    'but', 'if', 'their', 'which', 'you', 'me', 'my', 'our', 'your', 'they',
    'them', 'we', 'us', 'so', 'do', 'not', 'no', 'can', 'will', 'just'
}

# Basic suffixes for naive stemming
_SUFFIXES = (
    'ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment'
)

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def stem(word: str) -> str:
    """Very small stemming function removing common suffixes."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[:-len(suf)]
    return word


def tokenize(text: str) -> List[str]:
    """Tokenize a tweet and apply basic cleaning steps."""
    words = _TOKEN_RE.findall(text.lower())
    tokens = [stem(w) for w in words if len(w) >= 3 and w not in STOPWORDS]
    return tokens


def build_clean_corpus(texts: Iterable[str]) -> List[str]:
    """Return cleaned texts after tokenization and rejoin tokens."""
    return [' '.join(tokenize(t)) for t in texts]


def token_statistics(tokens: Iterable[str]) -> Dict[str, int]:
    """Compute total, unique and hapax tokens."""
    token_list = list(tokens)
    counter = Counter(token_list)
    total = len(token_list)
    unique = len(counter)
    once = sum(1 for c in counter.values() if c == 1)
    return {'total': total, 'unique': unique, 'once': once}


def generate_wordcloud(tokens: Iterable[str], output_file: str | None = None):
    """Generate a WordCloud from tokens if the package is available."""
    try:
        from wordcloud import WordCloud
    except ImportError as e:
        raise ImportError(
            "wordcloud package is not installed"
        ) from e

    wc = WordCloud(width=800, height=400).generate(' '.join(tokens))
    if output_file:
        wc.to_file(output_file)
    return wc
