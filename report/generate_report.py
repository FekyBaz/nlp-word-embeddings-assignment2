"""Generate a PDF report from report/generated/metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "report" / "generated" / "metrics.json"
FIGURES = ROOT / "report" / "generated" / "figures"
OUTPUT = ROOT / "report" / "report.pdf"


def build() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing {METRICS_PATH}. Run scripts/run_analysis.py first.")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    dataset = metrics["dataset"]
    results = metrics["results"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=19, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = [
        Paragraph("NLP Assignment 2: Word Embeddings Comparison", styles["CoverTitle"]),
        Paragraph("Word2Vec Skip-gram vs locally trained GloVe", styles["Heading2"]),
        Paragraph("Student ID: 2305467 | Natural Language Processing — Fall 2026", styles["BodyText"]),
        Spacer(1, 0.35 * cm),
        Paragraph("Objective and methodology", styles["Heading1"]),
        Paragraph(
            "This study compares 50-dimensional Word2Vec Skip-gram and a compact, locally implemented GloVe model "
            "on the assigned five-domain document classification dataset. Text is lowercased, tokenized into ASCII "
            "alphabetic words, filtered using scikit-learn's English stop-word list, and Porter-stemmed. A stratified "
            "80/20 split with random seed 42 is used. Embedding vocabularies are fitted only on training documents. "
            "Document vectors are means of available word vectors; a class-balanced Logistic Regression pipeline "
            "is evaluated on the held-out test set.", styles["BodyText"]),
        Spacer(1, 0.2 * cm),
        Paragraph("Dataset profile", styles["Heading1"]),
        Paragraph(
            f"Detected text column: {dataset['text_column']}; label column: {dataset['label_column']}. "
            f"Original rows: {dataset['original_rows']}; cleaned rows: {dataset['clean_rows']}; "
            f"removed rows: {dataset['dropped_rows']}. Training documents: {dataset['train_size']}; "
            f"test documents: {dataset['test_size']}.", styles["BodyText"]),
    ]
    class_data = [["Class", "Documents"]] + [[str(k), str(v)] for k, v in dataset["class_counts"].items()]
    class_table = Table(class_data, hAlign="LEFT", colWidths=[7 * cm, 3 * cm])
    class_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([class_table, Spacer(1, 0.3 * cm), Paragraph("Downstream evaluation", styles["Heading1"])])
    table_data = [["Embedding", "Vocab", "Train acc.", "Test acc.", "Macro F1", "Weighted F1", "Zero-vector test %"]]
    for item in results:
        table_data.append([
            item["model"], str(item["vocabulary_size"]), f"{item['train_accuracy']:.3f}",
            f"{item['test_accuracy']:.3f}", f"{item['macro_f1']:.3f}", f"{item['weighted_f1']:.3f}",
            f"{item['test_oov_document_rate']:.1%}",
        ])
    results_table = Table(table_data, repeatRows=1, colWidths=[4.0*cm, 1.1*cm, 1.25*cm, 1.15*cm, 1.1*cm, 1.35*cm, 1.55*cm])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([
        results_table, Spacer(1, 0.2 * cm),
        Paragraph("Test performance is the primary comparison. Training accuracy is included only as a diagnostic. "
                  "The zero-vector rate indicates test documents for which no token had an embedding.", styles["Small"]),
        Paragraph(
            f"Result summary: {max(results, key=lambda item: item['test_accuracy'])['model']} achieved the higher held-out "
            f"test accuracy ({max(results, key=lambda item: item['test_accuracy'])['test_accuracy']:.1%}) and macro-F1 "
            f"({max(results, key=lambda item: item['macro_f1'])['macro_f1']:.3f}) in this split. "
            "This is evidence for this dataset and configuration only, not a universal ranking.", styles["BodyText"]),
        Spacer(1, 0.2 * cm),
        PageBreak(),
        Paragraph("Confusion matrices", styles["Heading1"]),
    ])
    for item in results:
        safe_name = "word2vec" if item["model"].startswith("Word2Vec") else "glove"
        path = FIGURES / f"{safe_name}_confusion.png"
        if path.exists():
            story.extend([Paragraph(item["model"], styles["Heading2"]), Image(str(path), width=14*cm, height=10*cm), Spacer(1, 0.25*cm)])
    story.extend([PageBreak(), Paragraph("Per-class classification performance", styles["Heading1"])])
    story.append(Paragraph(
        "Precision is the proportion of predictions for a class that were correct; recall is the proportion of true "
        "members of that class recovered; F1 balances precision and recall. Class identifiers below are shown exactly "
        "as encoded in the source CSV (0–4); no semantic mapping to domain names is assumed.", styles["BodyText"]))
    for item in results:
        story.append(Paragraph(item["model"], styles["Heading2"]))
        report = item["per_class"]
        class_rows = [["Class", "Precision", "Recall", "F1-score", "Support"]]
        for class_name in dataset["labels"]:
            row = report[str(class_name)]
            class_rows.append([
                str(class_name), f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                f"{row['f1-score']:.3f}", str(int(row["support"])),
            ])
        for average_name, display_name in [("macro avg", "Macro avg."), ("weighted avg", "Weighted avg.")]:
            row = report[average_name]
            class_rows.append([
                display_name, f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                f"{row['f1-score']:.3f}", str(int(row["support"])),
            ])
        class_table = Table(class_rows, repeatRows=1, hAlign="LEFT", colWidths=[4.0*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.0*cm])
        class_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([class_table, Spacer(1, 0.25*cm)])
    story.extend([PageBreak(), Paragraph("Semantic similarity and visualization", styles["Heading1"])])
    for item in results:
        sem = item["semantic_similarity"]
        story.append(Paragraph(
            f"{item['model']}: {sem['evaluated_pairs']} of {sem['candidate_pairs']} curated word pairs "
            f"were available ({sem['coverage']:.1%} coverage). These cosine scores are illustrative only; "
            "the curated pairs are not a human-annotated benchmark.", styles["BodyText"]))
        for pair in sem["pairs"]:
            story.append(Paragraph(
                f"{pair['word_1']} / {pair['word_2']}: {pair['cosine_similarity']:.3f}", styles["Small"]))
        coherence = item["topic_coherence"]
        story.append(Paragraph(
            f"Training-only class-conditioned cohesion proxy (not standard NPMI): "
            f"{coherence['overall_mean_pairwise_cosine'] if coherence['overall_mean_pairwise_cosine'] is not None else 'not available'}.",
            styles["Small"]))
        for class_name, details in coherence["by_class"].items():
            score = details["mean_pairwise_cosine"]
            score_text = f"{score:.3f}" if score is not None else "not available"
            story.append(Paragraph(
                f"{class_name}: mean pairwise cosine={score_text}; selected terms: {', '.join(details['terms'][:8])}",
                styles["Small"]))
        story.append(Spacer(1, 0.15*cm))
    story.extend([
        Spacer(1, 0.2*cm),
        Paragraph("PCA and t-SNE", styles["Heading1"]),
        Paragraph("PCA is a linear projection that emphasizes high-variance directions. t-SNE emphasizes local "
                  "neighborhoods and can distort global distances. Both figures are exploratory and should not be "
                  "interpreted as definitive proof of semantic quality or topic coherence.", styles["BodyText"]),
    ])
    for name, label in [("word2vec", "Word2Vec Skip-gram"), ("glove", "GloVe")]:
        for method in ("pca", "tsne"):
            path = FIGURES / f"{name}_{method}.png"
            if path.exists():
                story.extend([Paragraph(f"{label} — {method.upper()}", styles["Heading3"]), Image(str(path), width=15*cm, height=9.5*cm)])
    story.extend([
        PageBreak(),
        Paragraph("Interpretation and limitations", styles["Heading1"]),
        Paragraph(
            f"On the fixed held-out test set, {max(results, key=lambda item: item['test_accuracy'])['model']} performed "
            f"best by accuracy ({max(results, key=lambda item: item['test_accuracy'])['test_accuracy']:.2%}) and macro-F1 "
            f"({max(results, key=lambda item: item['macro_f1'])['macro_f1']:.3f}). "
            f"The accuracy difference between the two methods is "
            f"{abs(results[0]['test_accuracy'] - results[1]['test_accuracy'])*100:.2f} percentage points. "
            "The per-class table should be consulted to see whether performance is uniform across classes. "
            "These metrics assess embeddings through mean document vectors and a class-balanced Logistic Regression "
            "classifier; they do not isolate embedding quality from all downstream design choices.", styles["BodyText"]),
        Spacer(1, 0.15*cm),
        Paragraph(
            "Semantic similarity: cosine similarities are reported for every curated pair found in both vocabularies. "
            "The list is small, manually selected, and not human-rated, so it provides examples rather than a validated "
            "semantic benchmark. Topic cohesion: the reported class-conditioned cosine score is an exploratory proxy, "
            "not standard NPMI; training labels guide term selection, so it is not an independent unsupervised measure.",
            styles["BodyText"]),
        Spacer(1, 0.15*cm),
        Paragraph(
            "Visualization: PCA is a linear projection that summarizes directions of high variance, while t-SNE is "
            "nonlinear and emphasizes local neighborhoods. Apparent clusters can differ because of these objectives; "
            "axes and distances in t-SNE should not be interpreted as global semantic distances. Neither plot alone "
            "proves semantic quality or topic coherence.", styles["BodyText"]),
        Spacer(1, 0.15*cm),
        Paragraph(
            "Limitations: the corpus is domain-specific; embeddings are trained locally with compact settings; "
            "document vectors are simple means; the curated pair list is limited; and results depend on the split "
            "and classifier. Labels are numeric in the supplied CSV, so this report preserves IDs 0–4 rather than "
            "guessing their mapping to politics, sport, tech, entertainment, and business. If the dataset documentation "
            "provides that mapping, it can be added after verification.", styles["BodyText"]),
        Spacer(1, 0.3*cm),
        Paragraph("Reproducibility", styles["Heading1"]),
        Paragraph(
            "Random seed: 42. Word2Vec vector size: 50, window: 5, min_count: 2, Skip-gram enabled. "
            f"GloVe epochs: {metrics['configuration']['glove_epochs']}. "
            "The notebook and scripts generate all reported metrics from the supplied CSV; values are not hard-coded.",
            styles["BodyText"]),
    ])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm, topMargin=1.5*cm, bottomMargin=1.5*cm, title="NLP Assignment 2 — Word Embeddings Comparison")
    doc.build(story)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    build()
