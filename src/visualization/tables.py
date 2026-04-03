"""LaTeX table generation for ACL-format research paper.

Every table loads values from JSON/CSV/YAML artifacts produced by the
pipeline -- zero hardcoded numbers.  Each ``generate_*`` function returns
a LaTeX string *and* writes a standalone ``.tex`` fragment that can be
included in the paper via ``\\input{tables/<name>.tex}``.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    """Load a JSON file and return the parsed dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_csv(path: str) -> List[Dict[str, str]]:
    """Load a CSV file and return a list of row dicts."""
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return the parsed dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _fmt(value: float, decimals: int = 2) -> str:
    """Format a float to *decimals* places."""
    return f"{value:.{decimals}f}"


def _fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a fraction (0-1) as a percentage string."""
    return f"{value * 100:.{decimals}f}\\%"


def _fmt_pm(mean: float, std: float, decimals: int = 2) -> str:
    """Format mean +/- std."""
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _bold(text: str) -> str:
    """Wrap text in \\textbf."""
    return f"\\textbf{{{text}}}"


def _sanitize_latex(text: str) -> str:
    """Escape characters that are special in LaTeX."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _write_tex(tex: str, output_path: str) -> None:
    """Write a LaTeX string to *output_path*, creating dirs as needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    logger.info("Wrote table to %s", output_path)


def _comma_int(value: int) -> str:
    """Format an integer with comma separators."""
    return f"{value:,}"


# ---------------------------------------------------------------------------
# Table 1 -- Dataset statistics
# ---------------------------------------------------------------------------

def generate_dataset_table(root_dir: str) -> str:
    """Table 1: Dataset statistics comparing English vs Arabic.

    Loads from ``results/analysis/dataset_statistics.json``.
    """
    stats_path = os.path.join(root_dir, "results", "analysis", "dataset_statistics.json")
    stats = _load_json(stats_path)

    rows = [
        ("Total utterances", _comma_int(stats["total_utterances"]),
         _comma_int(stats["total_utterances"])),
        ("Total sessions", _comma_int(stats["total_sessions"]),
         _comma_int(stats["total_sessions"])),
        ("Total words", _comma_int(stats["total_words_english"]),
         _comma_int(stats["total_words_arabic"])),
        ("Mean words/utterance", _fmt(stats["mean_words_english"]),
         _fmt(stats["mean_words_arabic"])),
        ("Teacher utterances", _comma_int(stats["speaker_counts"]["teacher"]),
         _comma_int(stats["speaker_counts"]["teacher"])),
        ("Student utterances", _comma_int(stats["speaker_counts"]["student"]),
         _comma_int(stats["speaker_counts"]["student"])),
        ("Word ratio (AR/EN)", "---",
         _fmt(stats["word_ratio_ar_en"], 3)),
    ]

    body_lines = [f"        {prop} & {en} & {ar} \\\\" for prop, en, ar in rows]
    body = "\n".join(body_lines)

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "\\textbf{Property} & \\textbf{English} & \\textbf{Arabic} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Dataset statistics for the English source corpus and "
        "its Arabic translation.}\n"
        "\\label{tab:dataset-statistics}\n"
        "\\end{table}\n"
    )

    out_path = os.path.join(root_dir, "paper", "tables", "dataset_statistics.tex")
    _write_tex(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
# Table 2 -- Translation quality metrics
# ---------------------------------------------------------------------------

def generate_translation_table(root_dir: str) -> str:
    """Table 2: Translation quality metrics.

    Loads from ``results/translation/mt_metrics.json``.
    """
    mt_path = os.path.join(root_dir, "results", "translation", "mt_metrics.json")
    mt = _load_json(mt_path)
    agg = mt["aggregate"]

    bleu_score = agg["bleu"]["score"]
    chrf_score = agg["chrf_pp"]["score"]
    meteor_mean = agg["meteor"]["mean"]
    meteor_std = agg["meteor"]["std"]
    bs = agg["bertscore"]

    rows = [
        ("BLEU", _fmt(bleu_score)),
        ("chrF++", _fmt(chrf_score)),
        ("METEOR", _fmt_pm(meteor_mean, meteor_std, 4)),
        ("BERTScore Precision", _fmt_pm(bs["precision_mean"], bs["precision_std"], 4)),
        ("BERTScore Recall", _fmt_pm(bs["recall_mean"], bs["recall_std"], 4)),
        ("BERTScore F1", _fmt_pm(bs["f1_mean"], bs["f1_std"], 4)),
    ]

    body_lines = [f"        {metric} & {val} \\\\" for metric, val in rows]
    body = "\n".join(body_lines)

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        "\\textbf{Metric} & \\textbf{Score} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Machine translation quality (NLLB-200). "
        "Values for METEOR and BERTScore are reported as mean "
        "$\\pm$ std across utterances.}\n"
        "\\label{tab:translation-quality}\n"
        "\\end{table}\n"
    )

    out_path = os.path.join(root_dir, "paper", "tables", "translation_quality.tex")
    _write_tex(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
# Table 3 -- Talk time by speaker role
# ---------------------------------------------------------------------------

def generate_talk_time_table(root_dir: str) -> str:
    """Table 3: Talk time distribution by speaker role.

    Loads from ``results/analysis/talk_time_by_speaker.csv``.
    """
    csv_path = os.path.join(root_dir, "results", "analysis", "talk_time_by_speaker.csv")
    rows_raw = _load_csv(csv_path)

    # Nicer display labels
    speaker_labels = {
        "teacher": "Teacher",
        "student": "Student",
        "multiple students": "Multiple Students",
    }

    body_lines: List[str] = []
    for row in rows_raw:
        speaker = speaker_labels.get(row["speaker"], row["speaker"])
        utt_count = _comma_int(int(row["utterance_count"]))
        words_en = _comma_int(int(row["total_words_en"]))
        words_ar = _comma_int(int(row["total_words_ar"]))
        mean_en = _fmt(float(row["mean_words_en"]))
        mean_ar = _fmt(float(row["mean_words_ar"]))
        pct_en = _fmt_pct(float(row["pct_words_en"]))
        pct_ar = _fmt_pct(float(row["pct_words_ar"]))
        body_lines.append(
            f"        {speaker} & {utt_count} & {words_en} & {words_ar} "
            f"& {mean_en} & {mean_ar} & {pct_en} & {pct_ar} \\\\"
        )
    body = "\n".join(body_lines)

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "\\textbf{Speaker} & \\textbf{Utts} & "
        "\\textbf{Words\\textsubscript{EN}} & \\textbf{Words\\textsubscript{AR}} & "
        "\\textbf{Mean\\textsubscript{EN}} & \\textbf{Mean\\textsubscript{AR}} & "
        "\\textbf{\\%\\textsubscript{EN}} & \\textbf{\\%\\textsubscript{AR}} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Talk time distribution by speaker role, comparing "
        "English source and Arabic translation.}\n"
        "\\label{tab:talk-time}\n"
        "\\end{table}\n"
    )

    out_path = os.path.join(root_dir, "paper", "tables", "talk_time.tex")
    _write_tex(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
# Table 4 -- Math content density
# ---------------------------------------------------------------------------

def generate_math_density_table(root_dir: str) -> str:
    """Table 4: Mathematical content detection summary.

    Loads from ``results/analysis/math_density_summary.json``.
    """
    md_path = os.path.join(root_dir, "results", "analysis", "math_density_summary.json")
    md = _load_json(md_path)

    rows = [
        ("Math lexicon size", _comma_int(md["lexicon_size"]), "---"),
        ("Total utterances", _comma_int(md["total_utterances"]),
         _comma_int(md["total_utterances"])),
        ("Utterances with math content",
         _comma_int(md["utterances_with_math_english"]),
         _comma_int(md["utterances_with_math_arabic"])),
        ("\\% with math content",
         _fmt_pct(md["pct_with_math_english"]),
         _fmt_pct(md["pct_with_math_arabic"])),
        ("Mean math density",
         _fmt(md["mean_math_density_english"], 4),
         _fmt(md["mean_math_density_arabic"], 4)),
    ]

    body_lines = [f"        {prop} & {en} & {ar} \\\\" for prop, en, ar in rows]
    body = "\n".join(body_lines)

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "\\textbf{Property} & \\textbf{English} & \\textbf{Arabic} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Mathematical content detection comparing English source "
        "and Arabic translation.}\n"
        "\\label{tab:math-density}\n"
        "\\end{table}\n"
    )

    out_path = os.path.join(root_dir, "paper", "tables", "math_density.tex")
    _write_tex(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
# Table 5 -- Main classification results
# ---------------------------------------------------------------------------

_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "dummy_most_frequent": "Majority",
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_svm": "TF-IDF + SVM",
    "mbert": "mBERT",
    "xlmr": "XLM-R",
    "arabert": "AraBERT",
}

_MODEL_ORDER: List[str] = [
    "dummy_most_frequent",
    "tfidf_lr",
    "tfidf_svm",
    "mbert",
    "xlmr",
    "arabert",
]

_TASK_DISPLAY: Dict[str, str] = {
    "focusing_questions": "FQ",
    "student_reasoning": "SR",
    "uptake": "UP",
}

_TASK_ORDER: List[str] = [
    "focusing_questions",
    "student_reasoning",
    "uptake",
]


def generate_classification_table(root_dir: str) -> str:
    """Table 5: Main classification results.

    Rows: models.  Columns: per-task accuracy and weighted F1 (mean +/- std).
    Best result per column is bolded.

    Loads from ``results/classification/all_results.json``.
    """
    res_path = os.path.join(root_dir, "results", "classification", "all_results.json")
    data = _load_json(res_path)

    # Collect values for bolding: best[task][metric] = (model_key, value)
    # metric in {"accuracy", "f1_weighted"}
    best: Dict[str, Dict[str, Tuple[str, float]]] = {
        task: {"accuracy": ("", -1.0), "f1_weighted": ("", -1.0)}
        for task in _TASK_ORDER
    }

    for task in _TASK_ORDER:
        for model in _MODEL_ORDER:
            if model not in data[task]:
                continue
            for metric in ("accuracy", "f1_weighted"):
                mean = data[task][model][metric]["mean"]
                if mean > best[task][metric][1]:
                    best[task][metric] = (model, mean)

    # Build rows
    body_lines: List[str] = []
    for model in _MODEL_ORDER:
        display = _MODEL_DISPLAY_NAMES[model]
        cells = [display]
        for task in _TASK_ORDER:
            if model not in data[task]:
                cells.extend(["---", "---"])
                continue
            for metric in ("accuracy", "f1_weighted"):
                mean = data[task][model][metric]["mean"]
                std = data[task][model][metric]["std"]
                cell = _fmt_pm(mean, std, 3)
                if best[task][metric][0] == model:
                    cell = _bold(cell)
                cells.append(cell)
        body_lines.append("        " + " & ".join(cells) + " \\\\")

    body = "\n".join(body_lines)

    # Column spec: model name + 2 cols (Acc, F1) per task = 1 + 6 = 7
    task_headers: List[str] = []
    for task in _TASK_ORDER:
        short = _TASK_DISPLAY[task]
        task_headers.append(
            f"\\multicolumn{{2}}{{c}}{{\\textbf{{{short}}}}}"
        )
    task_header_line = " & ".join([""] + task_headers) + " \\\\"

    metric_headers = ["\\textbf{Model}"]
    for _ in _TASK_ORDER:
        metric_headers.append("\\textbf{Acc}")
        metric_headers.append("\\textbf{F1\\textsubscript{w}}")
    metric_header_line = " & ".join(metric_headers) + " \\\\"

    # cmidrule lines
    cmidrule_lines: List[str] = []
    for i, _ in enumerate(_TASK_ORDER):
        start = 2 + i * 2
        end = start + 1
        cmidrule_lines.append(f"\\cmidrule(lr){{{start}-{end}}}")
    cmidrule = " ".join(cmidrule_lines)

    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{l" + "cc" * len(_TASK_ORDER) + "}\n"
        "\\toprule\n"
        f"{task_header_line}\n"
        f"{cmidrule}\n"
        f"{metric_header_line}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Classification results on Arabic data (5-fold cross-validation). "
        "Best result per column in \\textbf{bold}. "
        "FQ = Focusing Questions, SR = Student Reasoning, UP = Uptake.}\n"
        "\\label{tab:classification-results}\n"
        "\\end{table*}\n"
    )

    out_path = os.path.join(root_dir, "paper", "tables", "classification_results.tex")
    _write_tex(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
# Table 6 -- Cross-lingual transfer results
# ---------------------------------------------------------------------------

def generate_cross_lingual_table(root_dir: str) -> str:
    """Table 6: Cross-lingual transfer results.

    Show English validation accuracy/F1 and Arabic zero-shot accuracy/F1
    for mBERT and XLM-R across tasks.

    Loads from ``results/classification/cross_lingual_results.json``.
    """
    cl_path = os.path.join(
        root_dir, "results", "classification", "cross_lingual_results.json"
    )
    data = _load_json(cl_path)

    body_lines: List[str] = []
    for task in _TASK_ORDER:
        short = _TASK_DISPLAY[task]
        first_in_task = True
        for model_key in ("mbert", "xlmr"):
            display = _MODEL_DISPLAY_NAMES[model_key]
            entry = data[task][model_key]
            en_acc = _fmt(entry["english_validation"]["accuracy"], 4)
            en_f1 = _fmt(entry["english_validation"]["f1_weighted"], 4)
            ar_acc = _fmt(entry["arabic_zero_shot"]["accuracy"], 4)
            ar_f1 = _fmt(entry["arabic_zero_shot"]["f1_weighted"], 4)

            if first_in_task:
                row_label = f"\\multirow{{2}}{{*}}{{{short}}}"
                first_in_task = False
            else:
                row_label = ""

            body_lines.append(
                f"        {row_label} & {display} & {en_acc} & {en_f1} "
                f"& {ar_acc} & {ar_f1} \\\\"
            )
        body_lines.append("        \\midrule")
    # Remove trailing midrule
    body_lines = body_lines[:-1]

    body = "\n".join(body_lines)

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{llcccc}\n"
        "\\toprule\n"
        " & & \\multicolumn{2}{c}{\\textbf{EN (val)}} "
        "& \\multicolumn{2}{c}{\\textbf{AR (zero-shot)}} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n"
        "\\textbf{Task} & \\textbf{Model} & \\textbf{Acc} & "
        "\\textbf{F1\\textsubscript{w}} & \\textbf{Acc} & "
        "\\textbf{F1\\textsubscript{w}} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Cross-lingual transfer: models trained on English, "
        "evaluated on English validation and Arabic zero-shot.}\n"
        "\\label{tab:cross-lingual}\n"
        "\\end{table}\n"
    )

    out_path = os.path.join(root_dir, "paper", "tables", "cross_lingual_results.tex")
    _write_tex(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
# Table 7 -- Hyperparameters / training configuration
# ---------------------------------------------------------------------------

def generate_hyperparameter_table(root_dir: str) -> str:
    """Table 7: Training configuration / hyperparameters.

    Loads from ``config/config.yaml``.
    """
    cfg_path = os.path.join(root_dir, "config", "config.yaml")
    cfg = _load_yaml(cfg_path)

    # Transformer models
    transformer_names = {
        "AraBERT": cfg["models"]["arabert"],
        "mBERT": cfg["models"]["mbert"],
        "XLM-R": cfg["models"]["xlmr"],
    }

    # All three share most hyperparams; take from arabert as representative
    rep = cfg["models"]["arabert"]

    # Baselines
    tfidf_cfg = cfg["baselines"]["tfidf"]
    svm_cfg = cfg["baselines"]["svm"]
    lr_cfg = cfg["baselines"]["logistic_regression"]
    cls_cfg = cfg["classification"]

    ngram = f"({tfidf_cfg['ngram_range'][0]}, {tfidf_cfg['ngram_range'][1]})"

    rows = [
        ("\\multicolumn{2}{l}{\\textit{Transformer models}}", True),
        ("Max sequence length", str(rep["max_length"])),
        ("Learning rate", str(rep["learning_rate"])),
        ("Weight decay", str(rep["weight_decay"])),
        ("Warmup ratio", str(rep["warmup_ratio"])),
        ("Epochs (max)", str(rep["num_epochs"])),
        ("Batch size", str(rep["batch_size"])),
        ("Early stopping patience", str(rep["early_stopping_patience"])),
        ("Class-weighted loss", str(cls_cfg["class_weighted_loss"])),
        ("\\midrule", True),
        ("\\multicolumn{2}{l}{\\textit{Baseline models}}", True),
        ("TF-IDF max features", _comma_int(tfidf_cfg["max_features"])),
        ("TF-IDF n-gram range", ngram),
        ("TF-IDF sublinear TF", str(tfidf_cfg["sublinear_tf"])),
        ("SVM kernel", svm_cfg["kernel"]),
        ("SVM C", str(svm_cfg["C"])),
        ("LR C", str(lr_cfg["C"])),
        ("LR max iterations", _comma_int(lr_cfg["max_iter"])),
        ("\\midrule", True),
        ("\\multicolumn{2}{l}{\\textit{Evaluation}}", True),
        ("CV folds", str(cls_cfg["cv_folds"])),
        ("Test size", str(cls_cfg["test_size"])),
        ("Random seed", str(cfg["project"]["seed"])),
    ]

    body_lines: List[str] = []
    for item in rows:
        if isinstance(item[1], bool) and item[1]:
            # Section header or midrule
            if item[0] == "\\midrule":
                body_lines.append(f"        {item[0]}")
            else:
                body_lines.append(f"        {item[0]} \\\\")
        else:
            body_lines.append(f"        {item[0]} & {item[1]} \\\\")
    body = "\n".join(body_lines)

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{ll}\n"
        "\\toprule\n"
        "\\textbf{Hyperparameter} & \\textbf{Value} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Training configuration and hyperparameters.}\n"
        "\\label{tab:hyperparameters}\n"
        "\\end{table}\n"
    )

    out_path = os.path.join(root_dir, "paper", "tables", "hyperparameters.tex")
    _write_tex(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
# Table 8 -- Error analysis examples
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate a string and append ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def generate_error_analysis_table(root_dir: str) -> str:
    """Table 8: Error analysis examples per task.

    Loads from ``results/analysis/error_analysis.json``.

    Expected JSON format::

        {
          "focusing_questions": [
            {
              "arabic_text": "...",
              "english_text": "...",
              "true_label": 1,
              "predicted_label": 0,
              "confidence": 0.87
            }, ...
          ],
          "student_reasoning": [...],
          "uptake": [...]
        }

    Shows up to 2 examples per task.
    """
    ea_path = os.path.join(root_dir, "results", "analysis", "error_analysis.json")
    data = _load_json(ea_path)

    max_examples_per_task = 2

    body_lines: List[str] = []
    for task_idx, task in enumerate(_TASK_ORDER):
        short = _TASK_DISPLAY[task]
        task_data = data.get(task, [])
        # Support both list-of-dicts and dict-with-"examples"-key formats
        if isinstance(task_data, dict):
            examples = task_data.get("examples", [])[:max_examples_per_task]
        else:
            examples = list(task_data)[:max_examples_per_task]
        if not examples:
            continue

        for ex_idx, ex in enumerate(examples):
            ar_text = _sanitize_latex(_truncate(ex.get("text_arabic", ex.get("arabic_text", ""))))
            en_text = _sanitize_latex(_truncate(ex.get("text_english", ex.get("english_text", ""))))
            true_lbl = str(ex["true_label"])
            pred_lbl = str(ex["predicted_label"])
            conf = _fmt(ex["confidence"], 3)

            if ex_idx == 0:
                task_cell = f"\\multirow{{{len(examples)}}}{{*}}{{{short}}}"
            else:
                task_cell = ""

            body_lines.append(
                f"        {task_cell} & {ar_text} & {en_text} "
                f"& {true_lbl} & {pred_lbl} & {conf} \\\\"
            )
        if task_idx < len(_TASK_ORDER) - 1:
            body_lines.append("        \\midrule")

    body = "\n".join(body_lines)

    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{lp{4.2cm}p{4.2cm}ccc}\n"
        "\\toprule\n"
        "\\textbf{Task} & \\textbf{Arabic text} & \\textbf{English text} "
        "& \\textbf{True} & \\textbf{Pred} & \\textbf{Conf} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Representative misclassified examples per task. "
        "Arabic text is shown truncated.}\n"
        "\\label{tab:error-analysis}\n"
        "\\end{table*}\n"
    )

    out_path = os.path.join(root_dir, "paper", "tables", "error_analysis.tex")
    _write_tex(tex, out_path)
    return tex


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def generate_all_tables(root_dir: str) -> Dict[str, str]:
    """Generate all paper tables and save each as a .tex fragment.

    Parameters
    ----------
    root_dir : str
        Project root directory (e.g. ``/path/to/arabic-edu-convokit``).

    Returns
    -------
    dict
        Mapping from table name to its LaTeX string.
    """
    tables: Dict[str, str] = {}

    generators = [
        ("dataset_statistics", generate_dataset_table),
        ("translation_quality", generate_translation_table),
        ("talk_time", generate_talk_time_table),
        ("math_density", generate_math_density_table),
        ("classification_results", generate_classification_table),
        ("cross_lingual_results", generate_cross_lingual_table),
        ("hyperparameters", generate_hyperparameter_table),
    ]

    for name, func in generators:
        try:
            tables[name] = func(root_dir)
            logger.info("Generated table: %s", name)
        except FileNotFoundError:
            logger.warning(
                "Skipping table '%s': source data file not found.", name
            )
        except (KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping table '%s': data format error -- %s", name, exc
            )

    # Error analysis is optional (file may not exist yet)
    try:
        tables["error_analysis"] = generate_error_analysis_table(root_dir)
        logger.info("Generated table: error_analysis")
    except FileNotFoundError:
        logger.warning(
            "Skipping table 'error_analysis': "
            "results/analysis/error_analysis.json not found."
        )
    except (KeyError, json.JSONDecodeError) as exc:
        logger.warning(
            "Skipping table 'error_analysis': data format error -- %s", exc
        )

    logger.info(
        "Table generation complete: %d / %d tables produced.",
        len(tables),
        len(generators) + 1,
    )
    return tables
