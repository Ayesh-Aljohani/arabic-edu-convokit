"""
Visualization module for the Arabic Edu-ConvoKit project.

Generates publication-quality figures (300 DPI, seaborn-v0_8-whitegrid style)
for an NLP research paper on Arabic educational discourse classification.
All data is loaded from JSON/CSV result files -- nothing is hardcoded.

Each plotting function saves output to both:
    <root_dir>/results/figures/
    <root_dir>/paper/figures/
in PNG (300 DPI) and PDF formats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style constants
# ---------------------------------------------------------------------------
STYLE = "seaborn-v0_8-whitegrid"
DPI = 300
FONT_SIZE = 12

# Colorblind-friendly palette (IBM Design / Wong 2011)
COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "cyan":   "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "gray":   "#999999",
}

MODEL_COLORS = {
    "Majority":     COLORS["gray"],
    "TF-IDF + LR":  COLORS["cyan"],
    "TF-IDF + SVM": COLORS["blue"],
    "mBERT":        COLORS["orange"],
    "XLM-R":        COLORS["green"],
    "AraBERT":      COLORS["red"],
}

# Canonical ordering for models
MODEL_ORDER = [
    "Majority", "TF-IDF + LR", "TF-IDF + SVM",
    "mBERT", "XLM-R", "AraBERT",
]

# Internal key -> display name mapping for models
MODEL_KEY_TO_LABEL = {
    "dummy_most_frequent": "Majority",
    "tfidf_lr":            "TF-IDF + LR",
    "tfidf_svm":           "TF-IDF + SVM",
    "mbert":               "mBERT",
    "xlmr":                "XLM-R",
    "arabert":             "AraBERT",
}

# Internal key -> display name mapping for tasks
TASK_KEY_TO_LABEL = {
    "focusing_questions": "Focusing Questions",
    "student_reasoning":  "Student Reasoning",
    "uptake":             "Uptake",
}

TASK_ORDER = ["focusing_questions", "student_reasoning", "uptake"]

# Best model per task (for confusion matrices)
BEST_MODEL_PER_TASK = {
    "focusing_questions": "arabert",
    "student_reasoning":  "mbert",
    "uptake":             "xlmr",
}


def _apply_style() -> None:
    """Apply the global matplotlib style."""
    plt.style.use(STYLE)
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def _ensure_dirs(root_dir: Path) -> Tuple[Path, Path]:
    """Create and return both output directories."""
    results_fig = root_dir / "results" / "figures"
    paper_fig = root_dir / "paper" / "figures"
    results_fig.mkdir(parents=True, exist_ok=True)
    paper_fig.mkdir(parents=True, exist_ok=True)
    return results_fig, paper_fig


def _save_figure(
    fig: matplotlib.figure.Figure,
    results_dir: Path,
    paper_dir: Path,
    name: str,
) -> None:
    """Save figure to both output directories as PNG and PDF."""
    for directory in (results_dir, paper_dir):
        for ext in ("png", "pdf"):
            path = directory / f"{name}.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            logger.info("Saved %s", path)
    plt.close(fig)


# =========================================================================
# 1. Pipeline Architecture Diagram
# =========================================================================


def plot_pipeline_diagram(root_dir: Path) -> None:
    """Draw the pipeline architecture diagram using matplotlib patches/arrows."""
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Stage definitions: (label, x_center, width, color)
    stages = [
        ("NCTE\nEnglish Data",          1.0, 1.6, COLORS["cyan"]),
        ("NLLB-200\nTranslation",        3.0, 1.6, COLORS["blue"]),
        ("Arabic\nText",                 5.0, 1.6, COLORS["green"]),
        ("Preprocessing\n(pyarabic)",    7.0, 1.6, COLORS["orange"]),
        ("Feature Extraction\n(talk time +\nmath density)", 9.0, 1.8, COLORS["yellow"]),
        ("Classification\n(AraBERT, mBERT,\nXLM-R, baselines)", 11.2, 2.0, COLORS["red"]),
        ("Analysis",                    13.2, 1.2, COLORS["purple"]),
    ]

    box_height = 2.0
    y_center = 2.0

    for label, xc, w, color in stages:
        rect = mpatches.FancyBboxPatch(
            (xc - w / 2, y_center - box_height / 2),
            w, box_height,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(
            xc, y_center, label,
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="black",
            linespacing=1.3,
        )

    # Arrows between consecutive stages
    arrow_props = dict(
        arrowstyle="->,head_width=0.3,head_length=0.2",
        color="black", linewidth=1.5,
    )
    arrow_pairs = [
        (1.0, 3.0, 1.6, 1.6),
        (3.0, 5.0, 1.6, 1.6),
        (5.0, 7.0, 1.6, 1.6),
        (7.0, 9.0, 1.6, 1.8),
        (9.0, 11.2, 1.8, 2.0),
        (11.2, 13.2, 2.0, 1.2),
    ]
    for x1, x2, w1, w2 in arrow_pairs:
        ax.annotate(
            "",
            xy=(x2 - w2 / 2 - 0.05, y_center),
            xytext=(x1 + w1 / 2 + 0.05, y_center),
            arrowprops=arrow_props,
        )

    ax.set_title(
        "Arabic Edu-ConvoKit Processing Pipeline",
        fontsize=14, fontweight="bold", pad=20,
    )

    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "pipeline_diagram")
    logger.info("Pipeline diagram complete.")


# =========================================================================
# 2. Translation Quality
# =========================================================================


def plot_translation_quality(root_dir: Path) -> None:
    """Scatter plot of AR vs EN word counts and distribution of word-count ratios.

    Data is loaded from data/processed/all_sessions.csv.
    Translation metrics from results/translation/mt_metrics.json are shown
    as annotation text.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    csv_path = root_dir / "data" / "processed" / "all_sessions.csv"
    metrics_path = root_dir / "results" / "translation" / "mt_metrics.json"

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d utterances from %s", len(df), csv_path)

    en_words = df["talktime_words"].values
    ar_words = df["talktime_words_arabic"].values

    # Filter out zero-length utterances for ratio calculation
    mask = (en_words > 0) & (ar_words > 0)
    en_valid = en_words[mask]
    ar_valid = ar_words[mask]
    ratios = ar_valid / en_valid

    # Load aggregate MT metrics for annotation
    with open(metrics_path) as f:
        mt = json.load(f)
    agg = mt["aggregate"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # -- Panel A: scatter --
    ax = axes[0]
    ax.scatter(
        en_valid, ar_valid, alpha=0.25, s=12,
        color=COLORS["blue"], edgecolors="none",
    )

    # Correlation line (least-squares)
    coeffs = np.polyfit(en_valid, ar_valid, 1)
    x_fit = np.linspace(0, en_valid.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)
    r = np.corrcoef(en_valid, ar_valid)[0, 1]
    ax.plot(x_fit, y_fit, color=COLORS["red"], linewidth=2,
            label=f"Fit (r = {r:.3f})")

    # 1:1 reference line
    lim_max = max(en_valid.max(), ar_valid.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "--", color=COLORS["gray"],
            linewidth=1, label="1:1 reference")

    ax.set_xlabel("English Word Count")
    ax.set_ylabel("Arabic Word Count")
    ax.set_title("(a) EN vs AR Word Counts per Utterance")
    ax.legend(loc="upper left")

    # MT metrics text box
    bleu = agg["bleu"]["score"]
    chrf = agg["chrf_pp"]["score"]
    meteor = agg["meteor"]["mean"]
    bert_f1 = agg["bertscore"]["f1_mean"]
    metrics_text = (
        f"BLEU = {bleu:.1f}\n"
        f"chrF++ = {chrf:.1f}\n"
        f"METEOR = {meteor:.3f}\n"
        f"BERTScore F1 = {bert_f1:.3f}"
    )
    ax.text(
        0.98, 0.02, metrics_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.9),
    )

    # -- Panel B: ratio distribution --
    ax2 = axes[1]
    ax2.hist(
        ratios, bins=60, color=COLORS["green"], edgecolor="white",
        alpha=0.85, density=True,
    )
    median_ratio = np.median(ratios)
    ax2.axvline(
        median_ratio, color=COLORS["red"], linewidth=2, linestyle="--",
        label=f"Median = {median_ratio:.2f}",
    )
    ax2.axvline(
        1.0, color=COLORS["gray"], linewidth=1.5, linestyle=":",
        label="Ratio = 1.0",
    )
    ax2.set_xlabel("AR / EN Word Count Ratio")
    ax2.set_ylabel("Density")
    ax2.set_title("(b) Distribution of Word Count Ratios")
    ax2.legend(loc="upper right")
    ax2.set_xlim(0, min(ratios.max(), 5.0))

    fig.suptitle("Translation Quality Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "translation_quality")
    logger.info("Translation quality figure complete.")


# =========================================================================
# 3. Talk Time
# =========================================================================


def plot_talk_time(root_dir: Path) -> None:
    """Grouped bar chart of teacher/student/multiple word counts for EN and AR.

    Data is loaded from results/analysis/talk_time_by_speaker.csv.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    csv_path = root_dir / "results" / "analysis" / "talk_time_by_speaker.csv"
    df = pd.read_csv(csv_path)
    logger.info("Loaded talk-time stats from %s", csv_path)

    # Canonical speaker order
    speaker_order = ["teacher", "student", "multiple students"]
    df = df.set_index("speaker").reindex(speaker_order).reset_index()

    speakers = df["speaker"].str.title().tolist()
    en_total = df["total_words_en"].values
    ar_total = df["total_words_ar"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -- Panel A: total words --
    x = np.arange(len(speakers))
    width = 0.35
    ax = axes[0]
    bars_en = ax.bar(
        x - width / 2, en_total, width, label="English",
        color=COLORS["blue"], edgecolor="white",
    )
    bars_ar = ax.bar(
        x + width / 2, ar_total, width, label="Arabic",
        color=COLORS["orange"], edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(speakers)
    ax.set_ylabel("Total Word Count")
    ax.set_title("(a) Total Words by Speaker Role")
    ax.legend()
    ax.bar_label(bars_en, fmt="{:,.0f}", fontsize=8, padding=2)
    ax.bar_label(bars_ar, fmt="{:,.0f}", fontsize=8, padding=2)

    # -- Panel B: mean words per utterance --
    en_mean = df["mean_words_en"].values
    ar_mean = df["mean_words_ar"].values
    ax2 = axes[1]
    bars_en2 = ax2.bar(
        x - width / 2, en_mean, width, label="English",
        color=COLORS["blue"], edgecolor="white",
    )
    bars_ar2 = ax2.bar(
        x + width / 2, ar_mean, width, label="Arabic",
        color=COLORS["orange"], edgecolor="white",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(speakers)
    ax2.set_ylabel("Mean Words per Utterance")
    ax2.set_title("(b) Mean Words per Utterance by Speaker Role")
    ax2.legend()
    ax2.bar_label(bars_en2, fmt="{:.1f}", fontsize=8, padding=2)
    ax2.bar_label(bars_ar2, fmt="{:.1f}", fontsize=8, padding=2)

    fig.suptitle("Talk Time Analysis: English vs Arabic", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "talk_time")
    logger.info("Talk time figure complete.")


# =========================================================================
# 4. Math Density
# =========================================================================


def plot_math_density(root_dir: Path) -> None:
    """Side-by-side comparison of math detection rates and density for EN vs AR.

    Data is loaded from results/analysis/math_density_summary.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "analysis" / "math_density_summary.json"
    with open(json_path) as f:
        stats = json.load(f)
    logger.info("Loaded math-density stats from %s", json_path)

    fig, ax = plt.subplots(figsize=(6, 5))

    langs = ["English", "Arabic"]
    rates = [
        stats["pct_with_math_english"] * 100,
        stats["pct_with_math_arabic"] * 100,
    ]
    bar_colors = [COLORS["blue"], COLORS["orange"]]
    bars = ax.bar(langs, rates, color=bar_colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Utterances with Math Content (%)")
    ax.set_title("Math Content Detection Rate", fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.35)
    ax.bar_label(bars, fmt="{:.1f}%", fontsize=12, padding=3)

    n_en = stats["utterances_with_math_english"]
    n_ar = stats["utterances_with_math_arabic"]
    n_total = stats["total_utterances"]
    lexicon = stats.get("lexicon_size", "N/A")
    ax.text(
        0.98, 0.95,
        f"N = {n_total:,} utterances\nEN: {n_en:,} | AR: {n_ar:,}\nAR lexicon: {lexicon} terms",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.9),
    )
    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "math_density")
    logger.info("Math density figure complete.")


# =========================================================================
# 5. Classification Results (grouped bar chart)
# =========================================================================


def plot_classification_results(root_dir: Path) -> None:
    """Grouped bar chart of F1-weighted across all models and tasks.

    Data is loaded from results/classification/all_results.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "classification" / "all_results.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("Loaded classification results from %s", json_path)

    # Collect data for each task and model
    model_keys = list(MODEL_KEY_TO_LABEL.keys())
    task_labels = [TASK_KEY_TO_LABEL[t] for t in TASK_ORDER]
    n_tasks = len(TASK_ORDER)
    n_models = len(model_keys)

    means = np.zeros((n_tasks, n_models))
    stds = np.zeros((n_tasks, n_models))
    for ti, task_key in enumerate(TASK_ORDER):
        for mi, model_key in enumerate(model_keys):
            entry = data[task_key][model_key]["f1_weighted"]
            means[ti, mi] = entry["mean"]
            stds[ti, mi] = entry["std"]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_tasks)
    total_width = 0.75
    bar_width = total_width / n_models

    for mi, model_key in enumerate(model_keys):
        label = MODEL_KEY_TO_LABEL[model_key]
        offset = (mi - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means[:, mi],
            bar_width,
            yerr=stds[:, mi],
            label=label,
            color=MODEL_COLORS[label],
            edgecolor="white",
            capsize=2,
            error_kw={"linewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=FONT_SIZE)
    ax.set_ylabel("F1 Weighted")
    ax.set_title("Classification Performance Across Models and Tasks", fontweight="bold")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, frameon=True, fontsize=11,
    )
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "classification_results")
    logger.info("Classification results figure complete.")


# =========================================================================
# 6. Confusion Matrices
# =========================================================================


def plot_confusion_matrices(root_dir: Path) -> None:
    """3-panel figure with confusion matrices for the best model on each task.

    Loads per-fold predictions from individual *_results.json files and
    aggregates confusion matrices across folds.

    Best model per task:
        - Focusing Questions: AraBERT
        - Student Reasoning:  mBERT
        - Uptake:             XLM-R
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    class_dir = root_dir / "results" / "classification"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, task_key in enumerate(TASK_ORDER):
        model_key = BEST_MODEL_PER_TASK[task_key]
        fname = f"{task_key}_{model_key}_results.json"
        fpath = class_dir / fname
        with open(fpath) as f:
            res = json.load(f)
        logger.info("Loaded %s", fpath)

        # Aggregate confusion matrices across folds
        cm_agg = None
        for fold in res["per_fold"]:
            cm = np.array(fold["confusion_matrix"])
            if cm_agg is None:
                cm_agg = cm.copy()
            else:
                cm_agg += cm

        # Normalize for display (row-normalized = recall per class)
        cm_norm = cm_agg.astype(float) / cm_agg.sum(axis=1, keepdims=True)

        ax = axes[idx]
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        # Annotate cells with both count and percentage
        n_classes = cm_agg.shape[0]
        for i in range(n_classes):
            for j in range(n_classes):
                count = cm_agg[i, j]
                pct = cm_norm[i, j]
                text_color = "white" if pct > 0.5 else "black"
                ax.text(
                    j, i, f"{count}\n({pct:.1%})",
                    ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold",
                )

        class_labels = ["Negative", "Positive"]
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_labels)
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        model_label = MODEL_KEY_TO_LABEL[model_key]
        task_label = TASK_KEY_TO_LABEL[task_key]
        ax.set_title(f"{task_label}\n({model_label})", fontweight="bold")

    # Single colorbar for all panels
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Recall (row-normalized)")

    fig.suptitle(
        "Confusion Matrices for Best Models (Aggregated over 5 Folds)",
        fontsize=14, fontweight="bold", y=1.04,
    )
    _save_figure(fig, results_dir, paper_dir, "confusion_matrices")
    logger.info("Confusion matrices figure complete.")


# =========================================================================
# 7. Cross-Validation Box Plots
# =========================================================================


def plot_cv_boxplots(root_dir: Path) -> None:
    """Box plots showing 5-fold CV F1-weighted distributions per model/task.

    Data is loaded from results/classification/all_results.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "classification" / "all_results.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("Loaded classification results from %s", json_path)

    model_keys = list(MODEL_KEY_TO_LABEL.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ti, task_key in enumerate(TASK_ORDER):
        ax = axes[ti]
        box_data = []
        tick_labels = []
        colors = []
        for model_key in model_keys:
            values = data[task_key][model_key]["f1_weighted"]["values"]
            box_data.append(values)
            label = MODEL_KEY_TO_LABEL[model_key]
            tick_labels.append(label)
            colors.append(MODEL_COLORS[label])

        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.6),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor("black")

        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(TASK_KEY_TO_LABEL[task_key], fontweight="bold")
        if ti == 0:
            ax.set_ylabel("F1 Weighted")

    fig.suptitle(
        "5-Fold Cross-Validation Distributions",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "cv_boxplots")
    logger.info("CV box plots figure complete.")


# =========================================================================
# 8. Cross-Lingual Transfer
# =========================================================================


def plot_cross_lingual(root_dir: Path) -> None:
    """Grouped bar chart comparing EN-trained vs AR zero-shot for mBERT and XLM-R.

    Data is loaded from results/classification/cross_lingual_results.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "classification" / "cross_lingual_results.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("Loaded cross-lingual results from %s", json_path)

    cl_models = ["mbert", "xlmr"]
    cl_model_labels = {"mbert": "mBERT", "xlmr": "XLM-R"}
    conditions = ["english_validation", "arabic_zero_shot"]
    condition_labels = {"english_validation": "EN Validation", "arabic_zero_shot": "AR Zero-Shot"}
    condition_colors = {
        "english_validation": COLORS["blue"],
        "arabic_zero_shot": COLORS["orange"],
    }

    # One panel per task
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ti, task_key in enumerate(TASK_ORDER):
        ax = axes[ti]
        x = np.arange(len(cl_models))
        width = 0.3

        for ci, cond in enumerate(conditions):
            vals = []
            for model_key in cl_models:
                f1 = data[task_key][model_key][cond]["f1_weighted"]
                vals.append(f1)
            offset = (ci - 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                label=condition_labels[cond] if ti == 0 else None,
                color=condition_colors[cond],
                edgecolor="white",
            )
            ax.bar_label(bars, fmt="{:.3f}", fontsize=8, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels([cl_model_labels[m] for m in cl_models])
        ax.set_title(TASK_KEY_TO_LABEL[task_key], fontweight="bold")
        if ti == 0:
            ax.set_ylabel("F1 Weighted")
        ax.set_ylim(0, 1.15)

    # Single legend across all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, frameon=True,
    )

    fig.suptitle(
        "Cross-Lingual Transfer: English-Trained vs Arabic Zero-Shot",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "cross_lingual")
    logger.info("Cross-lingual figure complete.")


# =========================================================================
# 9. Session Word Count Heatmap
# =========================================================================


def plot_session_word_count_heatmap(root_dir: Path) -> None:
    """Heatmap of word counts per session and speaker role (EN and AR side-by-side).

    Rows = sessions (29), columns = speaker roles (teacher, student).
    Data is loaded from individual session CSVs in data/processed/.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    processed_dir = root_dir / "data" / "processed"
    session_files = sorted(
        [f for f in processed_dir.glob("*.csv") if f.stem.isdigit()],
        key=lambda f: int(f.stem),
    )

    roles = ["teacher", "student"]
    session_ids: List[str] = []
    en_matrix: List[List[int]] = []
    ar_matrix: List[List[int]] = []

    for sf in session_files:
        df = pd.read_csv(sf)
        session_ids.append(sf.stem)
        en_row: List[int] = []
        ar_row: List[int] = []
        for role in roles:
            subset = df[df["speaker"] == role]
            en_row.append(int(subset["talktime_words"].sum()))
            ar_row.append(int(subset["talktime_words_arabic"].sum()))
        en_matrix.append(en_row)
        ar_matrix.append(ar_row)

    logger.info("Loaded word counts for %d sessions.", len(session_ids))

    en_arr = np.array(en_matrix)
    ar_arr = np.array(ar_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharey=True)

    # -- Panel A: English --
    ax = axes[0]
    im_en = ax.imshow(en_arr, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels([r.title() for r in roles])
    ax.set_yticks(range(len(session_ids)))
    ax.set_yticklabels(session_ids)
    ax.set_ylabel("Session")
    ax.set_title("(a) English Word Count")
    for i in range(len(session_ids)):
        for j in range(len(roles)):
            val = en_arr[i, j]
            text_color = "white" if val > en_arr.max() * 0.6 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=8, color=text_color)
    fig.colorbar(im_en, ax=ax, shrink=0.6, label="Word Count")

    # -- Panel B: Arabic --
    ax2 = axes[1]
    im_ar = ax2.imshow(ar_arr, cmap="Oranges", aspect="auto")
    ax2.set_xticks(range(len(roles)))
    ax2.set_xticklabels([r.title() for r in roles])
    ax2.set_title("(b) Arabic Word Count")
    for i in range(len(session_ids)):
        for j in range(len(roles)):
            val = ar_arr[i, j]
            text_color = "white" if val > ar_arr.max() * 0.6 else "black"
            ax2.text(j, i, f"{val:,}", ha="center", va="center",
                     fontsize=8, color=text_color)
    fig.colorbar(im_ar, ax=ax2, shrink=0.6, label="Word Count")

    fig.suptitle(
        "Word Count by Session and Speaker Role",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "session_heatmap")
    logger.info("Session word count heatmap complete.")


# =========================================================================
# 10. Label Distribution
# =========================================================================


def plot_label_distribution(root_dir: Path) -> None:
    """3-panel bar chart showing positive vs negative counts for FQ, SR, UP.

    Data is loaded from results/analysis/dataset_statistics.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "analysis" / "dataset_statistics.json"
    with open(json_path) as f:
        stats = json.load(f)
    logger.info("Loaded dataset statistics from %s", json_path)

    label_dist = stats["label_distributions"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, task_key in enumerate(TASK_ORDER):
        ax = axes[idx]
        info = label_dist[task_key]
        positive = info["positive_count"]
        negative = info["labeled_count"] - positive
        counts = [negative, positive]
        labels = ["Negative", "Positive"]
        bar_colors = [COLORS["blue"], COLORS["orange"]]

        bars = ax.bar(labels, counts, color=bar_colors, edgecolor="white", width=0.5)
        ax.set_ylabel("Count")
        ax.set_title(TASK_KEY_TO_LABEL[task_key], fontweight="bold")
        ax.bar_label(bars, fmt="{:,.0f}", fontsize=10, padding=3)
        ax.set_ylim(0, max(counts) * 1.25)

        # Annotate positive rate
        rate = info["positive_rate"]
        ax.text(
            0.98, 0.95,
            f"Positive rate: {rate:.1%}\nN = {info['labeled_count']:,}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["gray"], alpha=0.9),
        )

    fig.suptitle(
        "Class Distribution by Task",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "label_distribution")
    logger.info("Label distribution figure complete.")


# =========================================================================
# 11. BERTScore (BLEU) per Session
# =========================================================================


def plot_bertscore_per_session(root_dir: Path) -> None:
    """Histogram of per-session BLEU score distribution.

    Data is loaded from results/translation/mt_metrics.json (per_session dict).
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "translation" / "mt_metrics.json"
    with open(json_path) as f:
        mt = json.load(f)
    logger.info("Loaded MT metrics from %s", json_path)

    per_session = mt["per_session"]
    bleu_scores = [per_session[s]["bleu"] for s in per_session]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(bleu_scores, bins=10, color=COLORS["blue"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("BLEU Score")
    ax.set_ylabel("Number of Sessions")
    ax.set_title("Distribution of Per-Session BLEU Scores", fontweight="bold")

    agg_bleu = mt["aggregate"]["bleu"]["score"]
    ax.axvline(
        agg_bleu, color=COLORS["red"], linewidth=2, linestyle="--",
        label=f"Aggregate = {agg_bleu:.1f}",
    )

    mean_bleu = np.mean(bleu_scores)
    std_bleu = np.std(bleu_scores)
    ax.text(
        0.97, 0.95,
        f"N = {len(bleu_scores)} sessions\nMean = {mean_bleu:.1f}\nStd = {std_bleu:.1f}\nRange: [{min(bleu_scores):.1f}, {max(bleu_scores):.1f}]",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["gray"], alpha=0.9),
    )
    ax.legend(loc="upper left", fontsize=11)

    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "bertscore_per_session")
    logger.info("BLEU distribution figure complete.")


# =========================================================================
# 12. Model Radar Chart
# =========================================================================


def plot_model_radar(root_dir: Path) -> None:
    """Radar chart with 3 axes (FQ, SR, UP accuracy), one polygon per model.

    Models shown: AraBERT, mBERT, XLM-R, TF-IDF+SVM.
    Data is loaded from results/classification/all_results.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "classification" / "all_results.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("Loaded classification results from %s", json_path)

    radar_models = ["arabert", "mbert", "xlmr", "tfidf_svm"]
    task_labels = [TASK_KEY_TO_LABEL[t] for t in TASK_ORDER]

    # Gather accuracy means
    model_values: Dict[str, List[float]] = {}
    for model_key in radar_models:
        vals: List[float] = []
        for task_key in TASK_ORDER:
            vals.append(data[task_key][model_key]["accuracy"]["mean"])
        model_values[model_key] = vals

    # Radar setup
    n_axes = len(TASK_ORDER)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for model_key in radar_models:
        label = MODEL_KEY_TO_LABEL[model_key]
        values = model_values[model_key] + model_values[model_key][:1]
        ax.plot(angles, values, "o-", linewidth=2, label=label,
                color=MODEL_COLORS[label], markersize=6)
        ax.fill(angles, values, alpha=0.1, color=MODEL_COLORS[label])

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        task_labels,
    )
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.set_title("Model Accuracy Across Tasks", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "model_radar")
    logger.info("Model radar figure complete.")


# =========================================================================
# 13. Positive-Class F1
# =========================================================================


def plot_positive_f1(root_dir: Path) -> None:
    """Grouped bar chart for positive-class F1 across all models and tasks.

    Similar to classification_results but showing F1_positive with error bars.
    Data is loaded from results/classification/all_results.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "classification" / "all_results.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("Loaded classification results from %s", json_path)

    model_keys = list(MODEL_KEY_TO_LABEL.keys())
    task_labels = [TASK_KEY_TO_LABEL[t] for t in TASK_ORDER]
    n_tasks = len(TASK_ORDER)
    n_models = len(model_keys)

    means = np.zeros((n_tasks, n_models))
    stds = np.zeros((n_tasks, n_models))
    for ti, task_key in enumerate(TASK_ORDER):
        for mi, model_key in enumerate(model_keys):
            entry = data[task_key][model_key]["f1_positive"]
            means[ti, mi] = entry["mean"]
            stds[ti, mi] = entry["std"]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_tasks)
    total_width = 0.75
    bar_width = total_width / n_models

    for mi, model_key in enumerate(model_keys):
        label = MODEL_KEY_TO_LABEL[model_key]
        offset = (mi - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means[:, mi],
            bar_width,
            yerr=stds[:, mi],
            label=label,
            color=MODEL_COLORS[label],
            edgecolor="white",
            capsize=2,
            error_kw={"linewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("F1 (Positive Class)")
    ax.set_title(
        "Positive-Class F1 Across Models and Tasks", fontweight="bold",
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, frameon=True, fontsize=11,
    )
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "positive_f1")
    logger.info("Positive F1 figure complete.")


# =========================================================================
# 14. Cross-Lingual Retention
# =========================================================================


def plot_cross_lingual_retention(root_dir: Path) -> None:
    """Bar chart of accuracy retention (AR_acc / EN_acc * 100) for mBERT and XLM-R.

    Includes horizontal reference line at 100%.
    Data is loaded from results/classification/cross_lingual_results.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "classification" / "cross_lingual_results.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("Loaded cross-lingual results from %s", json_path)

    cl_models = ["mbert", "xlmr"]
    cl_model_labels = {"mbert": "mBERT", "xlmr": "XLM-R"}
    task_labels = [TASK_KEY_TO_LABEL[t] for t in TASK_ORDER]
    n_tasks = len(TASK_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_tasks)
    width = 0.3

    for ci, model_key in enumerate(cl_models):
        retentions: List[float] = []
        for task_key in TASK_ORDER:
            en_acc = data[task_key][model_key]["english_validation"]["accuracy"]
            ar_acc = data[task_key][model_key]["arabic_zero_shot"]["accuracy"]
            retention = (ar_acc / en_acc) * 100 if en_acc > 0 else 0.0
            retentions.append(retention)
        offset = (ci - 0.5) * width
        label = cl_model_labels[model_key]
        bars = ax.bar(
            x + offset, retentions, width,
            label=label,
            color=MODEL_COLORS[label],
            edgecolor="white",
        )
        ax.bar_label(bars, fmt="{:.1f}%", fontsize=10, padding=3)

    ax.axhline(
        100, color=COLORS["red"], linewidth=1.5, linestyle="--",
        label="100% retention",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Accuracy Retention (%)")
    ax.set_title(
        "Cross-Lingual Accuracy Retention (AR / EN)",
        fontweight="bold",
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "cross_lingual_retention")
    logger.info("Cross-lingual retention figure complete.")


# =========================================================================
# 15. Error Rates
# =========================================================================


def plot_error_rates(root_dir: Path) -> None:
    """Bar chart of error rates for the best model per task.

    Data is loaded from results/analysis/error_analysis.json.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    json_path = root_dir / "results" / "analysis" / "error_analysis.json"
    with open(json_path) as f:
        data = json.load(f)
    logger.info("Loaded error analysis from %s", json_path)

    task_labels: List[str] = []
    error_rates: List[float] = []
    bar_labels: List[str] = []
    bar_colors: List[str] = []

    palette = [COLORS["red"], COLORS["orange"], COLORS["green"]]

    for idx, task_key in enumerate(TASK_ORDER):
        info = data[task_key]
        model_label = MODEL_KEY_TO_LABEL[info["model"]]
        task_labels.append(TASK_KEY_TO_LABEL[task_key])
        error_rates.append(info["error_rate"] * 100)
        bar_labels.append(model_label)
        bar_colors.append(palette[idx])

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(task_labels))
    bars = ax.bar(
        x, error_rates, color=bar_colors, edgecolor="white", width=0.5,
    )
    ax.bar_label(bars, fmt="{:.1f}%", fontsize=11, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{tl}\n({bl})" for tl, bl in zip(task_labels, bar_labels)],
    )
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Rates for Best Model per Task", fontweight="bold")
    ax.set_ylim(0, max(error_rates) * 1.25)

    # Annotate with counts
    for i, task_key in enumerate(TASK_ORDER):
        info = data[task_key]
        ax.text(
            i, error_rates[i] + max(error_rates) * 0.08,
            f"{info['n_errors']}/{info['n_total']}",
            ha="center", va="bottom", fontsize=9, color=COLORS["gray"],
        )

    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "error_rates")
    logger.info("Error rates figure complete.")


# =========================================================================
# 16. Word Count Ratio Violin
# =========================================================================


def plot_word_count_ratio_violin(root_dir: Path) -> None:
    """Violin plot of AR/EN word count ratio split by speaker role.

    Data is loaded from data/processed/all_sessions.csv.
    """
    _apply_style()
    results_dir, paper_dir = _ensure_dirs(root_dir)

    csv_path = root_dir / "data" / "processed" / "all_sessions.csv"
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d utterances from %s", len(df), csv_path)

    # Filter to valid ratios only
    mask = (df["talktime_words"] > 0) & (df["talktime_words_arabic"] > 0)
    df_valid = df.loc[mask].copy()
    df_valid["word_ratio"] = (
        df_valid["talktime_words_arabic"] / df_valid["talktime_words"]
    )

    # Cap extreme outliers for visualization
    cap = df_valid["word_ratio"].quantile(0.99)
    df_valid["word_ratio"] = df_valid["word_ratio"].clip(upper=cap)

    # Canonical speaker order with title case for display
    speaker_order = ["teacher", "student", "multiple students"]
    df_valid["speaker_display"] = df_valid["speaker"].str.title()
    display_order = [s.title() for s in speaker_order]

    fig, ax = plt.subplots(figsize=(9, 6))

    palette = {
        "Teacher": COLORS["blue"],
        "Student": COLORS["orange"],
        "Multiple Students": COLORS["green"],
    }

    sns.violinplot(
        data=df_valid,
        x="speaker_display",
        y="word_ratio",
        hue="speaker_display",
        order=display_order,
        hue_order=display_order,
        palette=palette,
        inner="quartile",
        linewidth=1.2,
        legend=False,
        ax=ax,
    )

    ax.axhline(
        1.0, color=COLORS["red"], linewidth=1.5, linestyle="--",
        label="Ratio = 1.0",
    )
    ax.set_xlabel("Speaker Role")
    ax.set_ylabel("AR / EN Word Count Ratio")
    ax.set_title(
        "Distribution of AR/EN Word Count Ratio by Speaker Role",
        fontweight="bold",
    )
    ax.legend(loc="upper right")

    # Annotate median per group
    for i, role in enumerate(speaker_order):
        subset = df_valid[df_valid["speaker"] == role]["word_ratio"]
        median_val = subset.median()
        ax.text(
            i, median_val + 0.02, f"med={median_val:.2f}",
            ha="center", va="bottom", fontsize=10, color=COLORS["black"],
        )

    fig.tight_layout()
    _save_figure(fig, results_dir, paper_dir, "word_ratio_violin")
    logger.info("Word count ratio violin figure complete.")


# =========================================================================
# 17. Master Function
# =========================================================================


def plot_all(root_dir: Path) -> None:
    """Generate all figures for the paper.

    Parameters
    ----------
    root_dir : Path
        Project root directory containing data/, results/, and paper/.
    """
    root_dir = Path(root_dir)
    logger.info("Generating all figures. Root: %s", root_dir)

    plot_pipeline_diagram(root_dir)
    plot_translation_quality(root_dir)
    plot_talk_time(root_dir)
    plot_math_density(root_dir)
    plot_classification_results(root_dir)
    plot_confusion_matrices(root_dir)
    plot_cv_boxplots(root_dir)
    plot_cross_lingual(root_dir)
    plot_session_word_count_heatmap(root_dir)
    plot_label_distribution(root_dir)
    plot_bertscore_per_session(root_dir)
    plot_model_radar(root_dir)
    plot_positive_f1(root_dir)
    plot_cross_lingual_retention(root_dir)
    plot_error_rates(root_dir)
    plot_word_count_ratio_violin(root_dir)

    logger.info("All figures generated successfully.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate all publication-quality figures for the Arabic Edu-ConvoKit paper."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory (default: inferred from script location).",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        choices=[
            "all", "pipeline", "translation", "talk_time",
            "math_density", "classification", "confusion",
            "cv_boxplots", "cross_lingual",
            "session_heatmap", "label_distribution",
            "bertscore_per_session", "model_radar",
            "positive_f1", "cross_lingual_retention",
            "error_rates", "word_ratio_violin",
        ],
        help="Which figure to generate (default: all).",
    )
    args = parser.parse_args()

    figure_funcs = {
        "pipeline":               plot_pipeline_diagram,
        "translation":            plot_translation_quality,
        "talk_time":              plot_talk_time,
        "math_density":           plot_math_density,
        "classification":         plot_classification_results,
        "confusion":              plot_confusion_matrices,
        "cv_boxplots":            plot_cv_boxplots,
        "cross_lingual":          plot_cross_lingual,
        "session_heatmap":        plot_session_word_count_heatmap,
        "label_distribution":     plot_label_distribution,
        "bertscore_per_session":  plot_bertscore_per_session,
        "model_radar":            plot_model_radar,
        "positive_f1":            plot_positive_f1,
        "cross_lingual_retention": plot_cross_lingual_retention,
        "error_rates":            plot_error_rates,
        "word_ratio_violin":      plot_word_count_ratio_violin,
    }

    if args.figure == "all":
        plot_all(args.root_dir)
    else:
        _apply_style()
        figure_funcs[args.figure](args.root_dir)
