"""Structural validation for generated Assignment 2 deliverables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "report" / "generated" / "metrics.json"
NOTEBOOK = ROOT / "assignment_2.ipynb"
PDF = ROOT / "report" / "report.pdf"
FIGURES = ROOT / "report" / "generated" / "figures"


def main() -> None:
    checks = 0
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)
            print(f"FAIL: {message}")
        else:
            print(f"PASS: {message}")

    check(METRICS.exists(), "metrics.json exists")
    if METRICS.exists():
        data = json.loads(METRICS.read_text(encoding="utf-8"))
        ds = data.get("dataset", {})
        results = data.get("results", [])
        check(ds.get("clean_rows", 0) > 0, "clean dataset is non-empty")
        check(ds.get("train_size", 0) + ds.get("test_size", 0) == ds.get("clean_rows", -1), "train/test sizes account for cleaned rows")
        check(len(ds.get("labels", [])) == 5, "dataset has five classes as specified")
        check(len(results) == 2, "exactly two embedding models are evaluated")
        check({r.get("model") for r in results} == {"Word2Vec Skip-gram", "GloVe (locally trained)"}, "both required embedding methods are present")
        for result in results:
            name = result.get("model", "unknown")
            check(0 <= result.get("test_accuracy", -1) <= 1, f"{name}: test accuracy is valid")
            check(0 <= result.get("macro_f1", -1) <= 1, f"{name}: macro F1 is valid")
            check(result.get("vocabulary_size", 0) > 0, f"{name}: vocabulary is non-empty")
            check(len(result.get("confusion_matrix", [])) == len(ds.get("labels", [])), f"{name}: confusion matrix has expected rows")
            check(result.get("semantic_similarity", {}).get("candidate_pairs", 0) > 0, f"{name}: semantic pair evaluation is documented")
        for name in ("word2vec_confusion.png", "glove_confusion.png", "word2vec_pca.png", "word2vec_tsne.png", "glove_pca.png", "glove_tsne.png"):
            check((FIGURES / name).exists(), f"figure exists: {name}")
    check(NOTEBOOK.exists(), "assignment notebook exists")
    if NOTEBOOK.exists():
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        check(any("Word2Vec" in cell.source and "GloVe" in cell.source for cell in notebook.cells if cell.cell_type == "markdown"), "notebook explains both models")
        errors_in_outputs = [
            (i, output.get("ename", "error"))
            for i, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        check(not errors_in_outputs, "executed notebook contains no error outputs")
        check(any(cell.cell_type == "code" and cell.execution_count is not None for cell in notebook.cells), "notebook has executed code cells")
    check(PDF.exists(), "report PDF exists")
    if PDF.exists():
        reader = PdfReader(str(PDF))
        check(len(reader.pages) >= 2, "PDF has at least two pages")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        check("Word Embeddings Comparison" in text, "PDF contains report title")
        check("2305467" in text, "PDF contains student ID")
        check("Word2Vec" in text and "GloVe" in text, "PDF discusses both models")
        check("Test acc." in text or "Test accuracy" in text, "PDF includes evaluation metrics")
    if errors:
        print(f"\nFAILED: {len(errors)} of {checks} checks.")
        sys.exit(1)
    print(f"\nPASS: {checks} validation checks.")


if __name__ == "__main__":
    main()
