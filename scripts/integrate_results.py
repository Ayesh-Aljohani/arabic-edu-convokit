"""Fill MARBERT, session-grouped, MT-divergent values into the paper LaTeX tables.

Reads completed result JSONs and rewrites the corresponding tex tables under
paper/the new paper/tables/. Idempotent and safe to re-run as more results
land.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "classification"
TABLES = ROOT / "paper" / "the new paper" / "tables"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fmt(mean, std, decimals=3):
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ----- Table 1: classification_results.tex (MARBERT row) ------------------

def update_classification_table():
    rows = []
    have_all = True
    for task in ["focusing_questions", "student_reasoning", "uptake"]:
        d = load(f"{task}_marbert_results.json")
        if d is None:
            logger.warning("MARBERT %s not yet done", task)
            have_all = False
            rows.append(("TBD", "TBD"))
        else:
            rows.append((
                fmt(d["accuracy"]["mean"], d["accuracy"]["std"]),
                fmt(d["f1_weighted"]["mean"], d["f1_weighted"]["std"]),
            ))
    fq_acc, fq_f1w = rows[0]; sr_acc, sr_f1w = rows[1]; up_acc, up_f1w = rows[2]
    # Build the row literal: end with TWO literal backslashes for LaTeX line break
    BS2 = chr(92) + chr(92)  # avoids any escape ambiguity
    new_row = (
        f"MARBERT v2 & {fq_acc} & {fq_f1w} & {sr_acc} & {sr_f1w} "
        f"& {up_acc} & {up_f1w} {BS2}"
    )

    path = TABLES / "classification_results.tex"
    text = path.read_text()
    import re
    # Match existing MARBERT row by line; use lambda to avoid \-escape interpretation
    text = re.sub(
        r"MARBERT v2 &[^\n]*",
        lambda m: new_row,
        text,
    )
    path.write_text(text)
    logger.info("Updated classification_results.tex: %s", new_row)
    return have_all


# ----- Table 2: classification_grouped.tex --------------------------------

def update_grouped_table():
    """Fill in session-grouped CV results vs. instance-stratified."""
    # Map (task, model_grouped_key) to (stratified_baseline_acc)
    strat = {
        ("focusing_questions", "arabert"): 0.893,
        ("focusing_questions", "mbert"): 0.871,
        ("focusing_questions", "xlmr"): 0.880,
        ("student_reasoning", "mbert"): 0.922,
        ("uptake", "xlmr"): 0.661,
    }
    pairs = [
        ("focusing_questions", "FQ", "AraBERT", "arabert", "arabert_grouped"),
        ("focusing_questions", "FQ", "mBERT", "mbert", "mbert_grouped"),
        ("focusing_questions", "FQ", "XLM-R", "xlmr", "xlmr_grouped"),
        ("student_reasoning", "SR", "mBERT", "mbert", "mbert_grouped"),
        ("uptake", "UP", "XLM-R", "xlmr", "xlmr_grouped"),
    ]
    rows = []
    have_all = True
    seen_tasks = set()
    for task, task_short, model_disp, model_strat, model_grouped in pairs:
        d = load(f"{task}_{model_grouped}_results.json")
        if d is None:
            have_all = False
            rows.append((task_short, model_disp, "TBD", strat[(task, model_strat)], "TBD"))
        else:
            grouped_acc = d["accuracy"]["mean"]
            grouped_std = d["accuracy"]["std"]
            base = strat[(task, model_strat)]
            delta = grouped_acc - base
            rows.append((
                task_short, model_disp,
                f"{grouped_acc:.3f} $\\pm$ {grouped_std:.3f}",
                f"{base:.3f}",
                f"{delta:+.3f}",
            ))

    # Build LaTeX rows; group multirows by task
    body = []
    by_task = {}
    for r in rows:
        by_task.setdefault(r[0], []).append(r)
    for task_short in ["FQ", "SR", "UP"]:
        if task_short not in by_task:
            continue
        sub = by_task[task_short]
        for i, r in enumerate(sub):
            multirow = (
                f"\\multirow{{{len(sub)}}}{{*}}{{{task_short}}} "
                if i == 0 else " "
            ) if len(sub) > 1 else f"{task_short} "
            body.append(
                f"{multirow}& {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\"
            )
        if task_short != "UP":
            body.append("\\midrule")

    table = (
        "\\begin{table}[t]\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{llccc}\n"
        "\\toprule\n"
        "\\textbf{Task} & \\textbf{Model} & \\textbf{Acc (grouped)} & "
        "\\textbf{Acc (stratified)} & $\\Delta$ Acc \\\\\n"
        "\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Session-grouped 5-fold cross-validation "
        "(\\texttt{GroupKFold}, with utterances of any one of the 29 NCTE "
        "sessions held out together) compared with the standard "
        "instance-stratified protocol. The drop $\\Delta$Acc estimates "
        "session-level leakage in the stratified protocol; small drops "
        "indicate the original numbers are robust to teacher-/session-"
        "specific signal.}\n"
        "\\label{tab:classification-grouped}\n"
        "\\end{table}\n"
    )
    path = TABLES / "classification_grouped.tex"
    path.write_text(table)
    logger.info("Updated classification_grouped.tex (have_all=%s)", have_all)
    return have_all


# ----- Table 3: mt_divergent_robustness.tex -------------------------------

def update_mt_divergent_table():
    p = RESULTS / "mt_divergent_results.json"
    if not p.exists():
        return False
    with open(p) as f:
        d = json.load(f)

    rows = []
    have_all = True
    for task, task_short in [("focusing_questions", "FQ"),
                              ("student_reasoning", "SR"),
                              ("uptake", "UP")]:
        for i, (model_key, model_disp) in enumerate([("mbert", "mBERT"), ("xlmr", "XLM-R")]):
            mr = d.get(task, {}).get(model_key, {})
            # New protocol: use the NLLB and Marian numbers from the SAME retrained classifier
            nllb_eval = mr.get("nllb_zero_shot", {})
            marian_eval = mr.get("marian_zero_shot", {})
            if not nllb_eval or "accuracy" not in nllb_eval:
                rows.append((task_short, model_disp,
                             "TBD", "TBD", "TBD", "TBD", i == 0))
                have_all = False
                continue
            nllb_acc = nllb_eval["accuracy"]
            if not marian_eval or "accuracy" not in marian_eval:
                rows.append((task_short, model_disp,
                             nllb_acc, "TBD", "TBD", "TBD", i == 0))
                have_all = False
                continue
            mar_acc = marian_eval["accuracy"]
            mar_f1w = marian_eval["f1_weighted"]
            d_acc = mar_acc - nllb_acc
            rows.append((task_short, model_disp,
                         nllb_acc, mar_acc, mar_f1w, d_acc, i == 0))

    body = []
    seen = {}
    for r in rows:
        ts = r[0]; seen.setdefault(ts, 0)
        seen[ts] += 1

    for r in rows:
        ts, model, nllb_acc, mar_acc, mar_f1w, d_acc, first = r
        if isinstance(mar_acc, str):
            multirow_cell = (
                f"\\multirow{{{seen[ts]}}}{{*}}{{{ts}}} " if first else " "
            )
            body.append(f"{multirow_cell}& {model} & {nllb_acc:.3f} & TBD & TBD & TBD \\\\")
        else:
            multirow_cell = (
                f"\\multirow{{{seen[ts]}}}{{*}}{{{ts}}} " if first else " "
            )
            body.append(
                f"{multirow_cell}& {model} & {nllb_acc:.3f} & {mar_acc:.3f} "
                f"& {mar_f1w:.3f} & {d_acc:+.3f} \\\\"
            )
        if first and ts != "UP" and seen[ts] == 1:
            pass
    # Add midrules between tasks
    out_lines = []
    last_task = None
    for line in body:
        # detect task change via multirow-cell content
        is_first = "\\multirow" in line
        if is_first and last_task is not None:
            out_lines.append("\\midrule")
        if is_first:
            ts_now = line.split("{")[2].rstrip("}").rstrip(" ")
            last_task = ts_now
        out_lines.append(line)

    table = (
        "\\begin{table}[t]\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{llcccc}\n"
        "\\toprule\n"
        "Task & Model & NLLB-200 Acc & MarianMT Acc & MarianMT F1$_{w}$ & $\\Delta$ Acc \\\\\n"
        "\\midrule\n"
        + "\n".join(out_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Cross-lingual classifier robustness across two "
        "independent MT systems. The same English-trained mBERT and XLM-R "
        "classifiers are evaluated zero-shot on (i) NLLB-200 and (ii) "
        "Helsinki-NLP/opus-mt-en-ar (MarianMT) translations of the labelled "
        "subset. $\\Delta$ Acc is MarianMT $-$ NLLB-200 (negative = lower "
        "on MarianMT). The classifiers are retrained for this experiment "
        "with $5$ epochs of early-stopped training, so the NLLB-200 column "
        "differs slightly from Table~\\ref{tab:cross-lingual}.}\n"
        "\\label{tab:mt-divergent}\n"
        "\\end{table}\n"
    )
    path = TABLES / "mt_divergent_robustness.tex"
    path.write_text(table)
    logger.info("Updated mt_divergent_robustness.tex (have_all=%s)", have_all)
    return have_all


def main():
    a = update_classification_table()
    b = update_grouped_table()
    c = update_mt_divergent_table()
    logger.info("Status: classification_full=%s grouped_full=%s mt_div_full=%s",
                a, b, c)


if __name__ == "__main__":
    main()
