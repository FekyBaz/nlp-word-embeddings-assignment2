# NLP Assignment 2 — Word Embeddings Comparison

Compare **Word2Vec Skip-gram** and **GloVe** embeddings on the assigned five-domain document-classification dataset. The project includes text preprocessing, PCA/t-SNE visualization, semantic word-pair inspection, a training-only topic-cohesion proxy, and held-out downstream classification evaluation.

## Assignment specification

See [NLP_anu_assignment2.pdf](NLP_anu_assignment2.pdf). Required deliverables are an executable Jupyter notebook with outputs/visualizations and a PDF report explaining methodology and findings.

## Dataset

The dataset CSV is committed at [`df_file.csv`](df_file.csv). The assignment source is [Kaggle: Text Document Classification Dataset](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset).

The pipeline detects common text/label column names at runtime and reports the actual columns and class counts. It fails loudly if the schema cannot be inferred. Verify the dataset's applicable terms before redistributing it outside this private coursework repository.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/inspect_dataset.py .
python scripts/run_analysis.py --data df_file.csv --output report/generated
python report/generate_report.py
python scripts/validate_outputs.py
```

To run the notebook top to bottom:

```powershell
jupyter nbconvert --to notebook --execute assignment_2.ipynb --output assignment_2_executed.ipynb --ExecutePreprocessor.timeout=1800
python report/generate_report.py
python scripts/validate_outputs.py
```

Open `assignment_2.ipynb` in Jupyter for the documented workflow. The CI workflow executes it in a clean environment and uploads the executed notebook, PDF report, metrics, and figures as a temporary 7-day artifact.

## Method and safeguards

- Fixed random seed: 42; stratified 80/20 train/test split.
- Preprocessing: lowercase, alphabetic tokenization, English stop-word removal, Porter stemming.
- Word2Vec: Gensim Skip-gram, 50 dimensions, context window 5, minimum count 2.
- GloVe: locally trained compact educational weighted least-squares implementation over co-occurrence counts, 50 dimensions and context window 5. This is **not** a pretrained published GloVe release.
- Both embedding vocabularies are fitted on training documents only. The downstream representation is the mean of available word vectors; evaluation uses class-balanced Logistic Regression on the same held-out split.
- Metrics include test accuracy, macro/weighted F1, per-class results, confusion matrices, vocabulary size, and the rate of test documents with no represented tokens.
- Semantic similarity uses a small curated set of illustrative word pairs and reports vocabulary coverage; it is not a gold-standard benchmark.
- Topic coherence is reported as mean pairwise cosine similarity among frequent class-associated training terms. This is an exploratory, label-informed proxy, **not** a standard NPMI topic-coherence metric.
- PCA and t-SNE plots are exploratory views; apparent clusters alone do not prove semantic quality.

## Repository contents

- `assignment_2.ipynb` — annotated notebook.
- `scripts/run_analysis.py` — reproducible model training and evaluation.
- `scripts/inspect_dataset.py` — dataset schema/quality inspection helper.
- `scripts/validate_outputs.py` — structural checks for generated metrics, figures, notebook, and PDF.
- `report/generate_report.py` — PDF report generator based on computed metrics.
- `.github/workflows/assignment2.yml` — clean-kernel notebook execution and artifact upload.

## Status and reproducibility

The code and CI workflow are in place. Metrics and PDF must be generated from the actual CSV and the workflow must pass before results are considered verified. No model-performance numbers are hard-coded in the report generator.
