"""Arabic tokenization utilities."""

import re


def tokenize_arabic(text: str) -> list[str]:
    """Tokenize Arabic text by matching Arabic character sequences."""
    if not isinstance(text, str):
        return []
    return re.findall(r"[\u0600-\u06FF]+", text)


def count_arabic_words(text: str) -> int:
    """Count Arabic words in text."""
    return len(tokenize_arabic(text))
