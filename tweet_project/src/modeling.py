import csv
from typing import Dict, List, Tuple

from .preprocessing import build_clean_corpus

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - si les packages sont manquants
    TfidfVectorizer = CountVectorizer = LogisticRegression = LinearSVC = None
    Pipeline = train_test_split = cross_val_score = None
    accuracy_score = precision_recall_fscore_support = None
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    from gensim.models import Word2Vec
except Exception:
    np = None
    Word2Vec = None


def load_texts_targets(path: str) -> tuple[list[str], list[int]]:
    """Charge le jeu de données et retourne les textes et les cibles entières."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    texts = [r["text"] for r in rows]
    y = [int(r["target"]) for r in rows]
    return texts, y


def build_vectorizer(name: str):
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn est requis pour les vectoriseurs")
    if name == "tfidf":
        return TfidfVectorizer()
    if name == "count":
        return CountVectorizer()
    raise ValueError("type de vectoriseur inconnu")


def build_classifier(name: str):
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn est requis pour les classificateurs")
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "svm":
        return LinearSVC()
    raise ValueError("type de modèle inconnu")


def train_evaluate(
    path: str,
    *,
    vectorizer: str = "tfidf",
    model: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
    ) -> Dict[str, float]:
    """Entraîne un pipeline de classification de texte et l'évalue."""
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


def train_predict(
    path: str,
    *,
    vectorizer: str = "tfidf",
    model: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[int], List[int]]:
    """Entraîne un pipeline et retourne les prédictions et les vraies étiquettes."""
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
    return list(y_pred), list(y_test)


def cross_validate(
    path: str,
    *,
    vectorizer: str = "tfidf",
    model: str = "logreg",
    cv: int = 5,
) -> float:
    """Effectue une validation croisée et retourne la précision moyenne."""
    texts, y = load_texts_targets(path)
    clean_texts = build_clean_corpus(texts, stem_words=False)

    vec = build_vectorizer(vectorizer)
    clf = build_classifier(model)
    pipe = Pipeline([("vect", vec), ("clf", clf)])
    scores = cross_val_score(pipe, clean_texts, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


if Word2Vec is not None and np is not None:

    def word2vec_features(texts: List[str], vector_size: int = 100):
        """Génère des caractéristiques Word2Vec en moyennant les vecteurs de mots."""
        tokenized = [t.split() for t in build_clean_corpus(texts, stem_words=False)]
        model = Word2Vec(tokenized, vector_size=vector_size, window=5, min_count=1, workers=1)

        def avg(tokens: List[str]):
            vecs = [model.wv[w] for w in tokens if w in model.wv]
            if not vecs:
                return np.zeros(vector_size)
            return np.mean(vecs, axis=0)

        return np.vstack([avg(tk) for tk in tokenized])
else:
    def word2vec_features(texts: List[str], vector_size: int = 100) -> None:
        """Fonction factice quand Word2Vec n'est pas disponible."""
        raise ImportError("gensim et numpy sont requis pour word2vec_features")


__all__ = [
    "train_evaluate",
    "train_predict",
    "cross_validate",
    "word2vec_features",
    "SKLEARN_AVAILABLE",
]
