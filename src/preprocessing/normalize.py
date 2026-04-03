"""Arabic text normalization using pyarabic."""

import re

from pyarabic import araby


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for NLP processing."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Strip diacritics (tashkeel)
    text = araby.strip_tashkeel(text)

    # Remove tatweel (kashida)
    text = araby.strip_tatweel(text)

    # Normalize alef variants to bare alef
    text = re.sub(r"[إأآا]", "ا", text)

    # Normalize taa marbuta to haa
    text = re.sub(r"ة", "ه", text)

    # Normalize alef maksura to yaa
    text = re.sub(r"ى", "ي", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
