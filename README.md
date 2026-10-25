# NLP Assignment 2 — Word Embeddings Comparison

Compare Word2Vec Skip-gram and GloVe embeddings on the Document Classification Dataset, including preprocessing, PCA/t-SNE visualization, semantic-similarity analysis, topic coherence, and downstream classification evaluation.

## Assignment deliverables

- A Jupyter notebook with code, outputs, visualizations, and explanations that runs top to bottom without errors.
- A PDF report summarizing methodology and key results.

Official assignment sheet: [NLP_anu_assignment2.pdf](NLP_anu_assignment2.pdf)

## Dataset

Required source: [Text Document Classification Dataset on Kaggle](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset)

The assignment specifies five domains: politics, sport, tech, entertainment, and business. Validate the actual downloaded files and schema before analysis; do not assume a row count or column names.

### Dataset archive currently uploaded

The uploaded `archive.zip` is in the repository root. Extract it locally into `data/raw/` after reviewing its contents and the dataset's applicable terms. Do not commit extracted third-party files until file sizes, schema, and license/terms have been checked.

On Windows PowerShell, from the repository root:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
Expand-Archive -LiteralPath .\archive.zip -DestinationPath .\data\raw -Force
py scripts/inspect_dataset.py data/raw
```

If downloading again from Kaggle, install and authenticate the Kaggle CLI using https://www.kaggle.com/settings/api, then run:

```powershell
py -m pip install kaggle
kaggle datasets download -d sunilthite/text-document-classification-dataset -p data/raw --unzip
py scripts/inspect_dataset.py data/raw
```

**Never commit Kaggle credentials, API tokens, or passwords.** The repository ignores raw data and credentials by default. Do not remove those safeguards without verifying dataset size and permission to redistribute.

## Development setup

Recommended: Python 3.11 or newer in a virtual environment.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Dependencies, notebook, validation scripts, and report will be finalized after the actual dataset schema and the GloVe training/loading approach have been verified.

## Reproducibility and evaluation principles

- Use a fixed random seed and record library versions and model configurations.
- Fit data-dependent preprocessing, vocabulary, and the downstream classifier only on the training split when evaluating generalization.
- Use the same split and a comparable evaluation protocol for both embeddings; report per-class metrics and a confusion matrix, not only accuracy.
- Document whether GloVe vectors are trained locally or loaded from a pretrained source.
- Treat PCA and t-SNE as exploratory visualizations, not proof of semantic quality.
- Report out-of-vocabulary coverage and explain limitations of the semantic-similarity and topic-coherence evaluation.

## Current status

Repository scaffolding and dataset-inspection utility are committed. Dataset contents have not yet been independently inspected in this environment; do not report model results until the data has been validated and the notebook executed.
