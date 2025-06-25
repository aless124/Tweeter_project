import csv
from collections import Counter
from typing import List, Dict, Tuple


def load_dataset(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Load CSV file and return list of rows and list of columns."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        columns = reader.fieldnames
    return rows, columns


def dataset_shape(rows: List[Dict[str, str]], columns: List[str]) -> Tuple[int, int]:
    """Return shape of the dataset (rows, columns)."""
    return len(rows), len(columns)


def missing_values(rows: List[Dict[str, str]], columns: List[str]) -> Dict[str, int]:
    missing = {col: 0 for col in columns}
    for row in rows:
        for col in columns:
            if row[col] == '' or row[col] is None:
                missing[col] += 1
    return missing


def duplicate_count(rows: List[Dict[str, str]], key: str = "id") -> int:
    ids = [row[key] for row in rows]
    return len(ids) - len(set(ids))


def text_length_stats(rows: List[Dict[str, str]], text_col: str = "text") -> Tuple[int, int, float]:
    lengths = [len(row[text_col]) for row in rows]
    return min(lengths), max(lengths), sum(lengths) / len(lengths)


def target_distribution(rows: List[Dict[str, str]], target_col: str = "target") -> Dict[str, int]:
    counter = Counter(row[target_col] for row in rows)
    return dict(counter)


def all_texts_non_empty(rows: List[Dict[str, str]], text_col: str = "text") -> bool:
    return all(row[text_col].strip() != "" for row in rows)


def unique_targets(rows: List[Dict[str, str]], target_col: str = "target") -> List[str]:
    return sorted(set(row[target_col] for row in rows))
