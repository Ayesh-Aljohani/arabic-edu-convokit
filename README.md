# Arabic Edu-ConvoKit

Adapting Educational Discourse Analysis to Arabic Classroom Conversations

## Abstract

Automated analysis of classroom discourse has shown promise for improving teaching quality, yet existing tools focus almost exclusively on English. We present **Arabic Edu-ConvoKit**, an open-source pipeline that adapts the Edu-ConvoKit framework to Arabic classroom conversations. Starting from 11,839 utterances across 29 mathematics lesson transcripts drawn from the NCTE corpus, we translate the data into Modern Standard Arabic using NLLB-200, achieving a BLEU score of 29.39, chrF++ of 47.07, and BERTScore F1 of 0.84. We benchmark three educational discourse tasks -- focusing question detection, student reasoning identification, and conversational uptake classification -- using AraBERT, mBERT, and XLM-RoBERTa alongside TF-IDF baselines. On Arabic data, AraBERT achieves the best accuracy of 89.3% for focusing questions with an F1 of 0.66, while mBERT leads on student reasoning (accuracy 92.2%, F1 0.77). Cross-lingual zero-shot transfer from English to Arabic retains 87.8-90.5% of source-language accuracy for focusing questions and student reasoning with XLM-RoBERTa, demonstrating feasible transfer even without Arabic training data. We release the full pipeline, translated dataset, Arabic mathematical vocabulary lexicon of 199 terms, and all trained models to support educational NLP research in Arabic-speaking contexts.

## Key Results

### Classification (5-Fold CV, Arabic Data)

| Task | Best Model | F1-weighted | F1-positive |
|---|---|---|---|
| Focusing Questions | AraBERT | 0.892 | 0.656 |
| Student Reasoning | mBERT | 0.921 | 0.765 |
| Uptake | XLM-R | 0.660 | 0.669 |

### Cross-Lingual Zero-Shot Transfer (English -> Arabic)

| Task | Model | F1-weighted | Accuracy Retention |
|---|---|---|---|
| Focusing Questions | XLM-R | 0.878 | 94.3% |
| Student Reasoning | XLM-R | 0.899 | 95.4% |
| Uptake | XLM-R | 0.665 | 98.9% |

### Translation Quality (NLLB-200, EN -> AR -> EN)

| Metric | Score |
|---|---|
| BLEU | 29.39 |
| chrF++ | 47.07 |
| METEOR | 0.531 |
| BERTScore F1 | 0.836 |

## Project Structure

```
arabic-edu-convokit/
├── config/
│   └── config.yaml                 # All hyperparameters and paths
├── data/
│   ├── raw/                        # Original English NCTE transcripts (29 sessions)
│   ├── translated/                 # Arabic translations (NLLB-200)
│   ├── back_translated/            # Back-translated English (for MT evaluation)
│   └── processed/                  # Normalized Arabic with features
├── src/
│   ├── translation/
│   │   ├── back_translate.py       # NLLB-200 EN->AR and AR->EN translation
│   │   └── mt_metrics.py           # BLEU, chrF++, METEOR, BERTScore
│   ├── preprocessing/
│   │   ├── normalize.py            # Arabic text normalization (pyarabic)
│   │   ├── tokenize_ar.py          # Arabic regex tokenizer
│   │   └── math_lexicon.py         # 199-term Arabic math vocabulary
│   ├── features/
│   │   ├── talk_time.py            # Talk time analysis by speaker role
│   │   └── math_density.py         # Mathematical content density
│   ├── classification/
│   │   ├── dataset.py              # PyTorch Dataset for utterances
│   │   ├── baselines.py            # Majority vote, TF-IDF + LR/SVM
│   │   ├── train.py                # Weighted HF Trainer, 5-fold stratified CV
│   │   ├── evaluate.py             # Metrics aggregation with 95% CI
│   │   └── cross_lingual.py        # Zero-shot EN->AR transfer
│   ├── analysis/
│   │   ├── cross_linguistic.py     # Cross-linguistic correlation analysis
│   │   ├── statistical_tests.py    # Bootstrap CI, Cohen's d, correlations
│   │   └── error_analysis.py       # Misclassification analysis
│   └── visualization/
│       ├── plots.py                # 16 publication-quality figures
│       └── tables.py               # 8 LaTeX table generators
├── scripts/
│   ├── run_backtranslation.py      # Phase 2: Translation pipeline
│   ├── run_preprocessing.py        # Phase 3: Arabic preprocessing
│   ├── run_experiments.py          # Phase 4: Classification experiments
│   └── run_analysis.py             # Phase 5: Analysis and visualization
├── results/
│   ├── classification/             # All model results as JSON
│   ├── translation/                # MT evaluation metrics
│   ├── analysis/                   # Statistical analysis outputs
│   └── figures/                    # Generated figures (PDF + PNG)
├── paper/
│   ├── main.tex                    # Full paper source
│   ├── main.pdf                    # Compiled paper
│   ├── references.bib              # Bibliography
│   ├── tables/                     # LaTeX table files
│   └── figures/                    # Paper figures (symlinked from results)
└── reference/
    └── educonvokit_paper.pdf       # Original Edu-ConvoKit paper
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Ayesh-Aljohani/arabic-edu-convokit.git
cd arabic-edu-convokit

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for METEOR)
python3 -c "import nltk; nltk.download('wordnet')"
```

## Reproduction

Run each phase sequentially from the project root:

```bash
# Phase 2: Back-translation (EN -> AR -> EN)
python3 -m scripts.run_backtranslation

# Phase 3: Arabic preprocessing and feature extraction
python3 -m scripts.run_preprocessing

# Phase 4: Classification experiments (5-fold CV + cross-lingual)
python3 -m scripts.run_experiments

# Phase 5: Analysis, figures, and tables
python3 -m scripts.run_analysis
```

Each script is resumable and will skip already-completed steps.

## Hardware Requirements

- **GPU**: Apple Silicon with MPS (tested on M4 Max) or NVIDIA GPU with CUDA
- **RAM**: ~48 GB recommended for transformer training
- **Disk**: ~10 GB for models and intermediate data
- **Time**: ~3 hours total on M4 Max (Phase 4 is the bottleneck)

For Apple Silicon, the pipeline automatically uses MPS acceleration. Set the following environment variable if needed:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Citation

```bibtex
@inproceedings{aljohani2025arabic,
  title={Arabic Edu-ConvoKit: Adapting Educational Discourse Analysis to Arabic Classroom Conversations},
  author={Aljohani, Ayesh and Manabri, Ahmed and Almetani, Mohammad and Mars, Mourad},
  year={2026},
  note={Umm Al-Qura University}
}
```

## License

See [LICENSE](LICENSE) for details.
