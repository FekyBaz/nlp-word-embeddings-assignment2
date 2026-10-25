# NLP Assignment 2 — Word Embeddings Comparison

Compare Word2Vec Skip-gram and GloVe embeddings on the Document Classification Dataset, with preprocessing, PCA/t-SNE visualization, semantic-similarity analysis, topic coherence, and downstream classification evaluation.

## Assignment deliverables

- A Jupyter notebook with code, outputs, visualizations, and explanations that runs top to bottom without errors.
- A PDF report summarizing methodology and key results.

Official assignment sheet: [NLP_anu_assignment2.pdf](NLP_anu_assignment2.pdf)

## Dataset

Required source: [Text Document Classification Dataset on Kaggle](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset)

The assignment describes five domains: politics, sport, tech, entertainment, and business. The data source is reported to contain 2,225 documents, but the actual downloaded files and their schema must be verified before analysis.

### Download locally

1. Install the Kaggle CLI:
   ```powershell
   py -m pip install kaggle
   ```
2. Authenticate with Kaggle using the official process at https://www.kaggle.com/settings/api. Keep credentials outside this repository. **Never commit `kaggle.json`, API tokens, or passwords.**
3. From the repository root, download and extract the dataset:
   ```powershell
   kaggle datasets download -d sunilthite/text-document-classification-dataset -p data/raw --unzip
   ```
4. Inspect the extracted files and confirm the text and label columns, row count, class labels, duplicates, missing values, and license/terms before using them.

The raw dataset is intentionally not committed yet. This avoids committing a dataset archive or republishing third-party data before verifying the files, size, and applicable terms. Once verified, decide whether to commit a permitted, reasonably sized data file or keep it locally and document the download procedure.

## Development setup

Recommended: Python 3.11 or newer in a virtual environment.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

A pinned `requirements.txt`, notebook, validation scripts, and report will be added after inspecting the real data schema and confirming the approach for GloVe.

## Reproducibility and evaluation principles

- Use a fixed seed and record versions/configuration.
- Fit any data-dependent preprocessing, vocabulary, and downstream classifier only on the training split when evaluating generalization.
- Use the same split and comparable evaluation protocol for embeddings; report class-wise metrics and a confusion matrix, not only accuracy.
- Document how GloVe vectors are obtained or trained; do not claim that it was trained locally if pretrained vectors are used.
- Treat PCA and t-SNE plots as exploratory visualizations, not proof of semantic quality.
- Record unavailable/OOV words and explain limitations of the similarity and topic-coherence evaluation.

## Repository status

Initial setup. Dataset files have not yet been validated or committed. Do not report results until they have been computed from the actual data and executed code.
