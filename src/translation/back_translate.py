"""Back-translate Arabic utterances to English using NLLB-200."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_nllb_model(
    model_name: str = "facebook/nllb-200-distilled-600M",
    device: str = "mps",
) -> tuple:
    """Load NLLB-200 model and tokenizer."""
    logger.info("Loading NLLB-200 model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device).eval()
    logger.info("Model loaded on %s", device)
    return model, tokenizer


def back_translate_batch(
    texts: list[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    src_lang: str = "arb_Arab",
    tgt_lang: str = "eng_Latn",
    max_length: int = 128,
    num_beams: int = 5,
    device: str = "mps",
) -> list[str]:
    """Translate a batch of Arabic texts to English."""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=128,
            num_beams=num_beams,
        )

    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


def back_translate_dataframe(
    df: pd.DataFrame,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    text_col: str = "text_arabic",
    batch_size: int = 8,
    device: str = "mps",
) -> pd.DataFrame:
    """Back-translate all Arabic text in a dataframe."""
    texts = df[text_col].tolist()
    all_translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Replace NaN/None with empty string
        batch = [t if isinstance(t, str) else "" for t in batch]
        translations = back_translate_batch(
            batch, model, tokenizer, device=device
        )
        all_translations.extend(translations)

        if (i // batch_size) % 10 == 0:
            logger.info(
                "Translated %d / %d utterances", len(all_translations), len(texts)
            )

    df = df.copy()
    df["text_back_translated"] = all_translations
    return df


def back_translate_all_sessions(
    translated_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "facebook/nllb-200-distilled-600M",
    batch_size: int = 8,
    device: str = "mps",
) -> dict[str, pd.DataFrame]:
    """Back-translate all session CSVs."""
    translated_dir = Path(translated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_nllb_model(model_name, device)

    csv_files = sorted(translated_dir.glob("*.csv"), key=lambda p: int(p.stem))
    results = {}

    for csv_file in csv_files:
        session_id = csv_file.stem
        output_path = output_dir / f"{session_id}.csv"

        if output_path.exists():
            logger.info("Session %s: already done, skipping", session_id)
            results[session_id] = pd.read_csv(output_path)
            continue

        logger.info("Back-translating session %s", session_id)
        df = pd.read_csv(csv_file)
        df = back_translate_dataframe(
            df, model, tokenizer, batch_size=batch_size, device=device
        )

        df.to_csv(output_path, index=False)
        results[session_id] = df

        logger.info(
            "Session %s: %d utterances back-translated", session_id, len(df)
        )

    return results
