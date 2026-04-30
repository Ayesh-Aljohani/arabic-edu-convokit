# Arabic Edu-ConvoKit

Adapting Educational Discourse Analysis to Arabic Classroom Conversations

> Status: companion code for a manuscript currently under review at *Data Intelligence* (Elsevier).

## Abstract

Automated classroom-discourse analysis enables instructional-quality measurement at scale, but its open-source toolchain has not been validated on Arabic. We present **Arabic Edu-ConvoKit**, the first publicly released pipeline adapting Edu-ConvoKit to Arabic mathematics classrooms. We translate 11,839 utterances from 29 NCTE elementary-mathematics sessions into Modern Standard Arabic with NLLB-200, characterising translations by round-trip preservation (BLEU 29.39, BERTScore-F1 0.836, word-count Pearson r = 0.874) and forward-direction LaBSE cosine (mean 0.854; 75 % above the 0.80 adequacy cutoff). We benchmark TF-IDF, AraBERT, mBERT, XLM-RoBERTa, and MARBERT under both stratified and session-grouped (one-teacher-out) five-fold cross-validation on three pedagogical tasks. AraBERT and MARBERT tie on focusing-questions (≈ 89.4 % accuracy); mBERT, AraBERT, and MARBERT are statistically indistinguishable on student-reasoning (~ 90 %, overlapping 95 % CIs); XLM-RoBERTa leads on uptake (66.1 ± 3.0 %). Session-grouped CV deltas are within [−0.9, +0.4] pp of the stratified split, so the headline numbers are not driven by teacher-level leakage. We then study cross-lingual robustness twice: (i) English-trained classifiers retain 91.2–98.9 % of source-language **accuracy** on NLLB-200 paraphrases, but minority-class F1 retention collapses to 40–46 % for mBERT on imbalanced FQ/SR while XLM-R holds 80–86 %, and both retain ≥ 96 % on uptake — we read this as *robustness to machine translation*, not full cross-lingual transfer, since both splits derive from the same English source; (ii) on independent MarianMT translations the retention profile is preserved (median Δ Acc + 0.85 pp). A zero-shot Qwen2.5-1.5B-Instruct LLM baseline trails the fine-tuned encoders by 13–26 pp on FQ/SR but exceeds fine-tuned XLM-R on uptake (+ 2.2 pp acc, + 5.6 pp on minority-class F1), suggesting that pragmatic-inference tasks may benefit from broader contextual reasoning. We release the pipeline, parallel corpus, 199-term Arabic mathematical lexicon, and all trained checkpoints.

## Key Results

### Classification — instance-stratified 5-fold CV on Arabic translations

| Task | Best model(s) | Accuracy | F1<sub>w</sub> | F1<sub>+</sub> |
|---|---|---|---|---|
| Focusing Questions | AraBERT, MARBERT-v2 (tied) | 0.893 / 0.894 | 0.892 / 0.894 | 0.656 / 0.669 |
| Student Reasoning | mBERT (CIs overlap with MARBERT, AraBERT) | 0.922 | 0.921 | 0.765 |
| Uptake | XLM-R | 0.661 | 0.659 | 0.659 |

### Session-grouped CV (one-teacher-out, GroupKFold over 29 sessions)

| Task | Model | Acc (grouped) | Acc (stratified) | Δ Acc |
|---|---|---|---|---|
| FQ | AraBERT | 0.890 ± 0.017 | 0.893 | −0.3 pp |
| FQ | mBERT | 0.872 ± 0.012 | 0.871 | +0.1 pp |
| FQ | XLM-R | 0.875 ± 0.014 | 0.880 | −0.5 pp |
| SR | mBERT | 0.913 ± 0.023 | 0.922 | −0.9 pp |
| UP | XLM-R | 0.665 ± 0.030 | 0.661 | +0.4 pp |

All deltas fall inside the per-fold ± 1.2–3.0 pp grouped-CV standard deviations — no statistically meaningful evidence of session-level leakage.

### Robustness to machine translation (English → Arabic, zero-shot)

| Task | Model | Acc retention | F1<sub>+</sub> retention |
|---|---|---|---|
| FQ | mBERT | 91.2 % | **45.9 %** |
| FQ | XLM-R | 94.3 % | 80.3 % |
| SR | mBERT | 92.4 % | **39.7 %** |
| SR | XLM-R | 95.4 % | 85.6 % |
| UP | mBERT | 96.2 % | 98.2 % |
| UP | XLM-R | 98.9 % | 96.2 % |

Accuracy retention obscures a sharp post-translation drop in minority-class F<sub>1</sub> on the imbalanced tasks for mBERT. We read these numbers as *robustness to machine translation*, not full cross-lingual transfer, because the Arabic test set is an NLLB-200 paraphrase of the same English source on which the classifier was trained.

### MT-system divergence (Helsinki MarianMT vs NLLB-200)

| Task | Model | NLLB-200 acc | MarianMT acc | Δ Acc |
|---|---|---|---|---|
| FQ | mBERT | 0.846 | 0.832 | −0.014 |
| FQ | XLM-R | 0.873 | 0.879 | +0.006 |
| SR | mBERT | 0.839 | 0.860 | +0.021 |
| SR | XLM-R | 0.861 | 0.879 | +0.018 |
| UP | mBERT | 0.649 | 0.660 | +0.011 |
| UP | XLM-R | 0.674 | 0.666 | −0.008 |

Median Δ Acc +0.85 pp, range [−1.4, +2.1] pp. Cross-system corpus BLEU between the two Arabic outputs ≈ 18, so the systems produce substantively different surface realisations on identical English input. The retention profile is preserved across MT pipelines.

### Zero-shot Qwen2.5-1.5B-Instruct LLM baseline (n = 200 stratified per task)

| Task | LLM acc | Best fine-tuned acc | Δ |
|---|---|---|---|
| FQ | 0.760 | 0.894 (AraBERT/MARBERT) | −13.4 pp |
| SR | 0.665 | 0.922 (mBERT) | −25.7 pp |
| UP | 0.683 | 0.661 (XLM-R) | **+2.2 pp** (LLM beats fine-tuned) |

On uptake the LLM also exceeds the fine-tuned XLM-R on minority-class F<sub>1</sub> (0.715 vs 0.659).

### Translation quality (NLLB-200 distilled-600M)

| Metric | Score |
|---|---|
| BLEU (round-trip) | 29.39 |
| chrF++ | 47.07 |
| METEOR | 0.531 ± 0.259 |
| BERTScore F<sub>1</sub> | 0.836 ± 0.090 |
| LaBSE forward cosine | 0.854 ± 0.098 |
| Fraction with LaBSE ≥ 0.80 | 74.9 % |

## Project Structure

```
arabic-edu-convokit/
├── config/                          # Hyperparameters and paths
├── data/
│   ├── raw/                         # Original English NCTE transcripts (29 sessions)
│   ├── translated/                  # Arabic translations (NLLB-200)
│   ├── back_translated/             # Back-translated English (for MT evaluation)
│   └── processed/
│       ├── all_sessions.csv         # Normalised Arabic with features
│       ├── all_sessions_with_sid.csv  # + explicit session_id column for GroupKFold
│       └── marian_ar_labeled.csv    # Independent Helsinki MarianMT translations
├── src/
│   ├── translation/                 # NLLB-200 EN↔AR, BLEU/chrF++/METEOR/BERTScore
│   ├── preprocessing/               # PyArabic normalisation, 199-term math lexicon
│   ├── features/                    # Talk-time and math-density analysis
│   ├── classification/              # Datasets, baselines, weighted Trainer, eval, X-lingual
│   ├── analysis/                    # Bootstrap CI, Cohen's d, correlations, error analysis
│   └── visualization/               # 16 publication-quality figures + 8 LaTeX tables
├── scripts/
│   ├── run_backtranslation.py       # NLLB-200 translation pipeline
│   ├── run_preprocessing.py         # Arabic preprocessing
│   ├── run_experiments.py           # 5-fold stratified CV (AraBERT, mBERT, XLM-R, baselines)
│   ├── run_analysis.py              # Statistical analysis and figures
│   ├── run_marbert.py               # MARBERT-v2 baseline
│   ├── run_session_grouped_cv.py    # GroupKFold(session_id) re-evaluation
│   ├── run_char_ngram_baseline.py   # char-(3,5) TF-IDF + LR/SVM
│   ├── run_comet_adequacy.py        # LaBSE forward-direction QE
│   ├── run_mt_divergent.py          # Helsinki MarianMT cross-system robustness
│   ├── run_arabic_llm.py            # Qwen2.5-1.5B-Instruct zero-shot baseline
│   ├── integrate_results.py         # Auto-fills LaTeX tables from result JSONs
│   └── auto_integrate.sh            # Watcher that recompiles on new result files
├── results/
│   ├── classification/              # Per-model JSONs + checkpoints (gitignored)
│   ├── translation/                 # mt_metrics.json, labse_qe.json
│   ├── analysis/                    # Statistical analysis outputs
│   └── figures/                     # Generated figures (PDF + PNG)
├── paper/
│   ├── figures/                     # Published figure exports (PDF, PNG, EPS)
│   └── JAISCR_submission.zip        # Archive of the prior JAISCR submission
└── reference/
    └── educonvokit_paper.pdf        # Original Edu-ConvoKit reference paper
```

The active manuscript LaTeX source is intentionally not included in this repository while the paper is under review.

## Installation

```bash
git clone https://github.com/Ayesh-Aljohani/arabic-edu-convokit.git
cd arabic-edu-convokit

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# NLTK data for METEOR
python3 -c "import nltk; nltk.download('wordnet')"
```

## Reproduction

Original pipeline (translation → preprocessing → main experiments → analysis):

```bash
python3 -m scripts.run_backtranslation     # NLLB-200 EN→AR→EN
python3 -m scripts.run_preprocessing       # PyArabic normalisation, features
python3 -m scripts.run_experiments         # 5-fold stratified CV + cross-lingual
python3 -m scripts.run_analysis            # Bootstrap CIs, figures, tables
```

Round-1–4 revision experiments (each runnable independently):

```bash
python scripts/run_marbert.py              # R4 — MARBERT-v2 baseline
python scripts/run_session_grouped_cv.py   # R5 — GroupKFold(session_id=29)
python scripts/run_char_ngram_baseline.py  # S1 — char-(3,5) TF-IDF + LR/SVM
python scripts/run_comet_adequacy.py       # R1 — LaBSE forward-direction QE
python scripts/run_mt_divergent.py         # R2 — Helsinki MarianMT divergence
python scripts/run_arabic_llm.py           # R4-partial — Qwen2.5 zero-shot
python scripts/integrate_results.py        # auto-fills LaTeX tables from JSONs
```

Each script is resumable and skips already-computed work.

## Hardware Requirements

- **GPU**: Apple Silicon with MPS (tested on M4 Max, 51 GB unified memory) or NVIDIA GPU with CUDA
- **RAM**: ~48 GB recommended for transformer training; 16 GB sufficient for the LLM zero-shot script (Qwen2.5-1.5B FP16)
- **Disk**: ~15 GB for models, MT outputs, and intermediate data
- **Time**: ~3 h for the original pipeline, +~2 h for the revision experiments end-to-end

```bash
# Apple Silicon convenience
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Citation

The manuscript is under review at *Data Intelligence*. Until it is published, please cite the repository directly via the **Cite this repository** button on GitHub (which uses [`CITATION.cff`](CITATION.cff)) or:

```bibtex
@misc{aljohani2026arabic,
  title  = {Arabic Edu-ConvoKit: Adapting Educational Discourse Analysis to Arabic Classroom Conversations},
  author = {Aljohani, Ayesh and Manabri, Ahmed and Almetani, Mohammad and Mars, Mourad},
  year   = {2026},
  note   = {Umm Al-Qura University; manuscript under review at \emph{Data Intelligence}},
  url    = {https://github.com/Ayesh-Aljohani/arabic-edu-convokit}
}
```

## License

See [LICENSE](LICENSE) for details.
