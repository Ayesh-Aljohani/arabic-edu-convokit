"""Zero-shot Arabic-LLM benchmark on the three pedagogical tasks.

Addresses reviewer R4 / NEW-1 in round 1: 'consider one Arabic LLM
(ALLaM/AceGPT/Fanar) zero/few-shot run'.

Strategy:
- Use AceGPT-7B-chat (FreedomIntelligence/AceGPT-7B-chat), an Arabic-tuned
  open-weights LLM, no gating, FP16 on Apple MPS (M4 Max, 51 GB unified RAM).
- Stratified sample of 200 utterances per task (600 total) to keep wall time
  reasonable on consumer hardware.
- Zero-shot prompting in Arabic, parsing 'YES'/'NO' (نعم/لا) responses.
- Report accuracy + minority-class F1 + weighted F1 vs the AraBERT/MARBERT
  baselines from Section 6.

Outputs:
  results/classification/arabic_llm_results.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "all_sessions.csv"
OUT_PATH = ROOT / "results" / "classification" / "arabic_llm_results.json"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_PER_TASK = 200
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_arabic_llm")


PROMPTS = {
    "focusing_questions": (
        "أنت مساعد متخصص في تحليل الخطاب التعليمي.\n"
        "مهمتك: تحديد ما إذا كان السؤال أدناه هو 'سؤال تركيز' (focusing question) "
        "أي سؤال يدفع الطالب إلى شرح تفكيره أو تبرير إجابته أو التوسع في فكرة "
        "(وليس مجرد سؤال يطلب جوابًا واحدًا صحيحًا أو غير صحيح).\n\n"
        "السؤال: \"{text}\"\n\n"
        "هل هذا سؤال تركيز؟ أجب بكلمة واحدة فقط: نعم أو لا."
    ),
    "student_reasoning": (
        "أنت مساعد متخصص في تحليل الخطاب التعليمي.\n"
        "مهمتك: تحديد ما إذا كانت عبارة الطالب أدناه تظهر 'تفكيرًا رياضيًا' "
        "(student reasoning)، أي أن الطالب يشرح كيف توصل إلى إجابته أو يبرر استنتاجه "
        "أو يربط بين أفكار رياضية، وليس مجرد ذكر إجابة عددية.\n\n"
        "العبارة: \"{text}\"\n\n"
        "هل تظهر هذه العبارة تفكيرًا رياضيًا؟ أجب بكلمة واحدة فقط: نعم أو لا."
    ),
    "uptake": (
        "أنت مساعد متخصص في تحليل الخطاب التعليمي.\n"
        "مهمتك: تحديد ما إذا كانت استجابة المعلم أدناه تستوعب 'استيعاب المحادثة' "
        "(uptake)، أي أن المعلم يبني بشكل واضح على ما قاله الطالب للتو "
        "(يكرر، يعيد صياغة، أو يطور فكرة الطالب)، وليس مجرد إقرار قصير أو الانتقال "
        "إلى موضوع جديد.\n\n"
        "السياق: قال الطالب: \"{prev_text}\"\n"
        "ثم قال المعلم: \"{text}\"\n\n"
        "هل تظهر استجابة المعلم استيعابًا واضحًا لما قاله الطالب؟ أجب بكلمة واحدة فقط: نعم أو لا."
    ),
}


def parse_yes_no(generated: str) -> int:
    """Map an LLM response to a binary label (1 = positive, 0 = negative)."""
    g = generated.strip().lower()
    # Arabic positive markers
    if any(k in g for k in ["نعم", "بلى", "أجل"]):
        return 1
    # English fallback (model may answer in English)
    if any(k in g for k in ["yes", "true", "correct"]):
        return 1
    # Arabic + English negatives
    if any(k in g for k in ["لا", "كلا", "no", "not", "false"]):
        return 0
    # Default: negative class (more conservative under uncertainty)
    return 0


def stratified_sample(df: pd.DataFrame, label_col: str, n: int, seed: int) -> pd.DataFrame:
    """Return n utterances stratified by label_col, preserving positive rate."""
    sub = df[df[label_col].notna()].copy()
    pos = sub[sub[label_col] == 1]
    neg = sub[sub[label_col] == 0]
    pos_rate = len(pos) / len(sub)
    n_pos = max(1, int(round(n * pos_rate)))
    n_neg = n - n_pos
    pos_s = pos.sample(min(n_pos, len(pos)), random_state=seed)
    neg_s = neg.sample(min(n_neg, len(neg)), random_state=seed)
    return pd.concat([pos_s, neg_s]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger.info("Loading dataset")
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d utterances", len(df))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading %s on %s (FP16)", MODEL_NAME, device)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_chat_template = hasattr(tok, "apply_chat_template") and tok.chat_template is not None
    logger.info("Using chat template: %s", use_chat_template)

    results: dict = {"model": MODEL_NAME, "n_per_task": N_PER_TASK, "tasks": {}}

    for task in ["focusing_questions", "student_reasoning", "uptake"]:
        logger.info("=== Task: %s ===", task)
        task_df = stratified_sample(df, task, N_PER_TASK, SEED)
        logger.info("Sampled %d (positive rate: %.3f)", len(task_df), task_df[task].mean())

        # For uptake, find previous turn from full df by index ordering
        if task == "uptake":
            full = df.reset_index(drop=False).rename(columns={"index": "orig_idx"})
            task_df = task_df.merge(
                full[["orig_idx", "text_arabic_normalized"]].rename(
                    columns={"text_arabic_normalized": "_curr"}
                ),
                left_on="text_arabic_normalized", right_on="_curr", how="left",
            ).drop_duplicates(subset=["text_arabic_normalized"]).head(N_PER_TASK)
            # Naive: prev_text = previous-row Arabic text in original df order
            text_to_prev: dict = {}
            arr = df["text_arabic_normalized"].fillna("").tolist()
            for i, t in enumerate(arr):
                text_to_prev[t] = arr[i - 1] if i > 0 else ""

        y_true = task_df[task].astype(int).tolist()
        y_pred: list = []
        raw_outputs: list = []

        prompt_tpl = PROMPTS[task]
        t1 = time.time()
        for i, row in task_df.iterrows():
            text_ar = (row.get("text_arabic_normalized") or row.get("text") or "")
            if task == "uptake":
                prev = text_to_prev.get(text_ar, "")
                user_msg = prompt_tpl.format(text=text_ar, prev_text=prev)
            else:
                user_msg = prompt_tpl.format(text=text_ar)

            if use_chat_template:
                msgs = [{"role": "user", "content": user_msg}]
                prompt = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            else:
                prompt = user_msg

            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            label = parse_yes_no(decoded)
            y_pred.append(label)
            raw_outputs.append(decoded[:60])

            if (i + 1) % 25 == 0 or i == len(task_df) - 1:
                rate = (i + 1) / (time.time() - t1)
                logger.info("  %d/%d (%.1f/s)", i + 1, len(task_df), rate)

        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cls_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        results["tasks"][task] = {
            "n": len(task_df),
            "accuracy": acc,
            "f1_weighted": f1w,
            "f1_positive": f1pos,
            "f1_macro": f1macro,
            "classification_report": cls_rep,
            "sample_outputs": raw_outputs[:5],
        }
        logger.info(
            "%s: acc=%.3f, f1_w=%.3f, f1_+=%.3f", task, acc, f1w, f1pos,
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", OUT_PATH)


if __name__ == "__main__":
    sys.exit(main())
