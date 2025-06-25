import csv
from typing import Dict, List

from text_preprocessing import build_clean_corpus
import dataset_analysis

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except Exception as e:  # pragma: no cover - if packages missing
    TfidfVectorizer = CountVectorizer = LogisticRegression = LinearSVC = None
    Pipeline = train_test_split = cross_val_score = None
    accuracy_score = precision_recall_fscore_support = None

try:
    import numpy as np
    from gensim.models import Word2Vec
except Exception:
    np = None
    Word2Vec = None


def load_texts_targets(path: str) -> tuple[list[str], list[int]]:
    """Load dataset and return texts and integer targets."""
    rows, _ = dataset_analysis.load_dataset(path)
    texts = [r["text"] for r in rows]
    y = [int(r["target"]) for r in rows]
    return texts, y


def build_vectorizer(name: str):
    if name == "tfidf":
        return TfidfVectorizer()
    if name == "count":
        return CountVectorizer()
    raise ValueError("unknown vectorizer type")


def build_classifier(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "svm":
        return LinearSVC()
    raise ValueError("unknown model type")


def train_evaluate(
    path: str,
    *,
    vectorizer: str = "tfidf",
    model: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    """Train a text classification pipeline and evaluate it."""
    texts, y = load_texts_targets(path)
    clean_texts = build_clean_corpus(texts, stem_words=False)

    vec = build_vectorizer(vectorizer)
    clf = build_classifier(model)

    X = vec.fit_transform(clean_texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def cross_validate(
    path: str,
    *,
    vectorizer: str = "tfidf",
    model: str = "logreg",
    cv: int = 5,
) -> float:
    """Perform cross validation and return mean accuracy."""
    texts, y = load_texts_targets(path)
    clean_texts = build_clean_corpus(texts, stem_words=False)

    vec = build_vectorizer(vectorizer)
    clf = build_classifier(model)
    pipe = Pipeline([("vect", vec), ("clf", clf)])
    scores = cross_val_score(pipe, clean_texts, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


if Word2Vec is not None and np is not None:

    def word2vec_features(texts: List[str], vector_size: int = 100) -> np.ndarray:
        tokenized = [t.split() for t in build_clean_corpus(texts, stem_words=False)]
        model = Word2Vec(tokenized, vector_size=vector_size, window=5, min_count=1, workers=1)

        def avg(tokens: List[str]) -> np.ndarray:
            vecs = [model.wv[w] for w in tokens if w in model.wv]
            if not vecs:
                return np.zeros(vector_size)
            return np.mean(vecs, axis=0)

        return np.vstack([avg(tk) for tk in tokenized])


__all__ = [
    "train_evaluate",
    "cross_validate",
    "word2vec_features",
]
