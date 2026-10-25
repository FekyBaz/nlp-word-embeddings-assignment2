"""Run the Assignment 2 embedding comparison pipeline.

Usage:
    python scripts/run_analysis.py --data df_file.csv --output report/generated

The loader detects common text/label column names and fails loudly if it cannot
identify them. GloVe is trained locally with weighted least-squares SGD over
word co-occurrence counts; Word2Vec is trained with Gensim Skip-gram.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
VECTOR_SIZE = 50
WINDOW = 5
MIN_COUNT = 2
STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()

TEXT_CANDIDATES = ("text", "document", "content", "data", "sentence", "article", "news", "body")
LABEL_CANDIDATES = ("label", "category", "class", "topic", "target", "type")


def load_dataset(path: Path) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Place df_file.csv in the repository root or pass --data.")
    df = pd.read_csv(path)
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"Dataset must contain at least two columns; got shape={df.shape}.")
    normalized = {str(c).strip().lower(): c for c in df.columns}

    def choose(candidates: tuple[str, ...], excluded: set[str]) -> str | None:
        for candidate in candidates:
            if candidate in normalized and normalized[candidate] not in excluded:
                return normalized[candidate]
        return None

    label_col = choose(LABEL_CANDIDATES, set())
    # Common exported dataframe index columns are not useful features.
    usable = [c for c in df.columns if not str(c).strip().lower().startswith("unnamed:")]
    if label_col is None:
        candidates = []
        for col in usable:
            n_unique = df[col].nunique(dropna=True)
            if 2 <= n_unique <= min(30, max(2, len(df) // 10)):
                candidates.append((n_unique, col))
        if candidates:
            label_col = min(candidates, key=lambda x: x[0])[1]
    if label_col is None:
        raise ValueError(f"Cannot infer label column from {list(df.columns)}. Set a recognized name such as label/category/class.")
    text_col = choose(TEXT_CANDIDATES, {label_col})
    if text_col is None:
        candidates = []
        for col in usable:
            if col == label_col:
                continue
            sample = df[col].dropna().astype(str).head(100)
            if len(sample) and sample.str.len().mean() > 20 and sample.nunique() > 1:
                candidates.append((sample.str.len().mean(), col))
        if candidates:
            text_col = max(candidates, key=lambda x: x[0])[1]
    if text_col is None or text_col == label_col:
        raise ValueError(f"Cannot infer text column from {list(df.columns)}. Set a recognized name such as text/content/document.")
    return df, str(text_col), str(label_col)


def preprocess(text: object) -> list[str]:
    if pd.isna(text):
        return []
    words = re.findall(r"[a-z]+", str(text).lower())
    return [STEMMER.stem(w) for w in words if w not in STOP_WORDS and len(w) > 1]


def train_glove(
    sentences: list[list[str]], vector_size: int = VECTOR_SIZE, window: int = WINDOW,
    min_count: int = MIN_COUNT, epochs: int = 25, learning_rate: float = 0.025,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], float]]:
    """Train a compact GloVe model using weighted least-squares SGD.

    This is a transparent educational implementation, not a claim of parity with
    large published/pretrained GloVe releases.
    """
    counts = Counter(token for sentence in sentences for token in sentence)
    vocab = [word for word, count in counts.items() if count >= min_count]
    vocab.sort()
    if len(vocab) < 2:
        raise ValueError("Too few vocabulary terms for GloVe; inspect the dataset and preprocessing.")
    index = {word: i for i, word in enumerate(vocab)}
    cooc: dict[tuple[int, int], float] = defaultdict(float)
    for sentence in sentences:
        ids = [index[w] for w in sentence if w in index]
        for i, center in enumerate(ids):
            left = max(0, i - window)
            right = min(len(ids), i + window + 1)
            for j in range(left, right):
                if i != j:
                    cooc[(center, ids[j])] += 1.0 / abs(i - j)
    pairs = np.array(list(cooc.keys()), dtype=np.int32)
    values = np.array(list(cooc.values()), dtype=np.float64)
    if len(values) == 0:
        raise ValueError("No co-occurrence pairs were produced.")
    rng = np.random.default_rng(SEED)
    n = len(vocab)
    w = rng.normal(0, 0.1, (n, vector_size))
    c = rng.normal(0, 0.1, (n, vector_size))
    bw = np.zeros(n, dtype=np.float64)
    bc = np.zeros(n, dtype=np.float64)
    # AdaGrad accumulators keep the simple SGD stable.
    gw = np.full_like(w, 1e-8)
    gc = np.full_like(c, 1e-8)
    gbw = np.full_like(bw, 1e-8)
    gbc = np.full_like(bc, 1e-8)
    weights = np.minimum(1.0, (values / 100.0) ** 0.75)
    targets = np.log(values)
    for epoch in range(epochs):
        order = rng.permutation(len(values))
        for start in range(0, len(order), 1024):
            batch = order[start:start + 1024]
            ii, jj = pairs[batch, 0], pairs[batch, 1]
            wi, cj = w[ii].copy(), c[jj].copy()
            diff = np.sum(wi * cj, axis=1) + bw[ii] + bc[jj] - targets[batch]
            grad = 2.0 * weights[batch] * diff
            grad_w = grad[:, None] * cj
            grad_c = grad[:, None] * wi
            np.add.at(gw, ii, grad_w * grad_w)
            np.add.at(gc, jj, grad_c * grad_c)
            np.add.at(gbw, ii, grad * grad)
            np.add.at(gbc, jj, grad * grad)
            np.add.at(w, ii, -learning_rate * grad_w / np.sqrt(gw[ii]))
            np.add.at(c, jj, -learning_rate * grad_c / np.sqrt(gc[jj]))
            np.add.at(bw, ii, -learning_rate * grad / np.sqrt(gbw[ii]))
            np.add.at(bc, jj, -learning_rate * grad / np.sqrt(gbc[jj]))
        if not np.isfinite(w).all() or not np.isfinite(c).all():
            raise FloatingPointError(f"GloVe training diverged at epoch {epoch + 1}.")
    vectors = w + c
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    return {word: vectors[i].astype(np.float32) for i, word in enumerate(vocab)}, cooc_to_words(cooc, vocab)


def cooc_to_words(cooc: dict[tuple[int, int], float], vocab: list[str]) -> dict[tuple[str, str], float]:
    return {(vocab[i], vocab[j]): value for (i, j), value in cooc.items()}


def model_vectors(model: Word2Vec) -> dict[str, np.ndarray]:
    vectors = model.wv.vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    return {word: vectors[i] for i, word in enumerate(model.wv.index_to_key)}


def document_vectors(token_docs: list[list[str]], vectors: dict[str, np.ndarray], dim: int) -> np.ndarray:
    result = np.zeros((len(token_docs), dim), dtype=np.float32)
    for i, tokens in enumerate(token_docs):
        found = [vectors[t] for t in tokens if t in vectors]
        if found:
            result[i] = np.mean(found, axis=0)
    return result


def evaluate_embeddings(
    name: str, vectors: dict[str, np.ndarray], train_docs: list[list[str]],
    test_docs: list[list[str]], y_train: np.ndarray, y_test: np.ndarray, labels: list[str],
) -> tuple[dict, np.ndarray]:
    dim = len(next(iter(vectors.values())))
    x_train = document_vectors(train_docs, vectors, dim)
    x_test = document_vectors(test_docs, vectors, dim)
    # The embedding model is trained on training documents only. Empty vectors are
    # retained as zeros so every test example remains represented in the metrics.
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    report = classification_report(y_test, pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0)
    result = {
        "model": name,
        "vocabulary_size": len(vectors),
        "train_accuracy": float(accuracy_score(y_train, classifier.predict(x_train))),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "test_oov_document_rate": float(np.mean(np.linalg.norm(x_test, axis=1) == 0)),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=np.arange(len(labels))).tolist(),
    }
    return result, pred


def curated_similarity_pairs() -> list[tuple[str, str]]:
    # Domain-relevant examples; report coverage rather than silently dropping OOV pairs.
    return [
        ("government", "minister"), ("election", "vote"), ("president", "politic"),
        ("team", "player"), ("game", "match"), ("goal", "season"),
        ("technology", "computer"), ("software", "system"), ("internet", "web"),
        ("film", "actor"), ("music", "song"), ("show", "movie"),
        ("market", "company"), ("business", "profit"), ("bank", "stock"),
    ]


def semantic_similarity(vectors: dict[str, np.ndarray]) -> dict:
    rows = []
    for left, right in curated_similarity_pairs():
        if left in vectors and right in vectors:
            sim = float(np.dot(vectors[left], vectors[right]) / (np.linalg.norm(vectors[left]) * np.linalg.norm(vectors[right]) + 1e-12))
            rows.append({"word_1": left, "word_2": right, "cosine_similarity": sim})
    return {"evaluated_pairs": len(rows), "candidate_pairs": len(curated_similarity_pairs()), "coverage": len(rows) / len(curated_similarity_pairs()), "pairs": rows}


def plot_confusion(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Number of documents")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_embedding_space(vectors: dict[str, np.ndarray], title: str, prefix: Path, top_n: int = 100) -> None:
    # Use most frequent vocabulary words (already ordered in Word2Vec; sort by vector key for GloVe).
    words = list(vectors)[:top_n]
    matrix = np.vstack([vectors[w] for w in words])
    if len(words) < 3:
        warnings.warn(f"Skipping PCA/t-SNE for {title}: only {len(words)} words available.")
        return
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.7)
    for word, (x, y) in zip(words, coords):
        ax.annotate(word, (x, y), fontsize=7, alpha=0.8)
    ax.set_title(f"{title}: PCA of word vectors")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_pca.png"), dpi=160)
    plt.close(fig)
    # Perplexity must be lower than sample count.
    perplexity = min(30, max(2, (len(words) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=SEED)
    coords = tsne.fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.7)
    for word, (x, y) in zip(words, coords):
        ax.annotate(word, (x, y), fontsize=7, alpha=0.8)
    ax.set_title(f"{title}: t-SNE of word vectors (perplexity={perplexity})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_tsne.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("df_file.csv"))
    parser.add_argument("--output", type=Path, default=Path("report/generated"))
    parser.add_argument("--glove-epochs", type=int, default=25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw, text_col, label_col = load_dataset(args.data)
    before = len(raw)
    clean = raw[[text_col, label_col]].copy()
    clean[text_col] = clean[text_col].fillna("").astype(str)
    clean[label_col] = clean[label_col].astype("string").str.strip()
    clean = clean[(clean[text_col].str.strip() != "") & clean[label_col].notna() & (clean[label_col] != "")]
    clean = clean.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    clean["tokens"] = clean[text_col].map(preprocess)
    clean = clean[clean["tokens"].map(bool)].reset_index(drop=True)
    if clean[label_col].nunique() < 2:
        raise ValueError("At least two non-empty classes are required.")
    class_counts = clean[label_col].value_counts().sort_index().to_dict()
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        clean[text_col].tolist(), clean[label_col].astype(str).to_numpy(), test_size=0.2,
        random_state=SEED, stratify=clean[label_col].astype(str).to_numpy()
    )
    # Preprocess train/test separately. Embedding vocabularies are learned from training docs only.
    train_docs = [preprocess(x) for x in x_train_raw]
    test_docs = [preprocess(x) for x in x_test_raw]
    train_docs = [x for x in train_docs if x]
    # Preserve one vector row per split sample; empty docs are possible after preprocessing.
    train_docs = [preprocess(x) for x in x_train_raw]
    test_docs = [preprocess(x) for x in x_test_raw]
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    labels = list(encoder.classes_)
    w2v = Word2Vec(
        sentences=train_docs, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT,
        sg=1, workers=1, seed=SEED, epochs=30, negative=5, sample=1e-3
    )
    w2v_vectors = model_vectors(w2v)
    glove_vectors, _ = train_glove(train_docs, epochs=args.glove_epochs)
    results = []
    predictions = {}
    for name, vectors in (("Word2Vec Skip-gram", w2v_vectors), ("GloVe (locally trained)", glove_vectors)):
        metrics, pred = evaluate_embeddings(name, vectors, train_docs, test_docs, y_train, y_test, labels)
        metrics["semantic_similarity"] = semantic_similarity(vectors)
        results.append(metrics)
        predictions[name] = pred.tolist()
        safe_name = "word2vec" if name.startswith("Word2Vec") else "glove"
        plot_confusion(metrics["confusion_matrix"], labels, f"{name} — test confusion matrix", figure_dir / f"{safe_name}_confusion.png")
        plot_embedding_space(vectors, name, figure_dir / safe_name)
    output = {
        "dataset": {
            "path": str(args.data), "original_rows": before, "clean_rows": len(clean),
            "dropped_rows": before - len(clean), "text_column": text_col, "label_column": label_col,
            "class_counts": class_counts, "labels": labels, "train_size": len(y_train),
            "test_size": len(y_test), "split_seed": SEED, "test_fraction": 0.2,
            "preprocessing": "lowercase; retain ASCII alphabetic tokens; remove sklearn English stop words; Porter stemming",
        },
        "configuration": {"vector_size": VECTOR_SIZE, "window": WINDOW, "min_count": MIN_COUNT, "seed": SEED, "glove_epochs": args.glove_epochs},
        "results": results,
        "notes": [
            "Word2Vec and GloVe vocabularies are fitted using training documents only.",
            "Document embeddings are the mean of available word vectors.",
            "The downstream classifier is StandardScaler + class-balanced LogisticRegression.",
            "The curated word-pair similarities are illustrative and not a gold-standard benchmark; coverage is reported.",
            "PCA and t-SNE are exploratory views of the most frequent/first vocabulary terms, not supervised topic clusters.",
            "The local GloVe implementation is educational and is not comparable in scale to published pretrained GloVe vectors.",
        ],
        "test_predictions": predictions,
    }
    output_path = args.output / "metrics.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = pd.DataFrame([{k: r[k] for k in ("model", "vocabulary_size", "train_accuracy", "test_accuracy", "macro_f1", "weighted_f1", "test_oov_document_rate")} for r in results])
    summary.to_csv(args.output / "model_comparison.csv", index=False)
    print(json.dumps({"dataset": output["dataset"], "summary": summary.to_dict(orient="records"), "metrics_path": str(output_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
