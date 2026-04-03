"""Mathematical content density computation."""

import pandas as pd

from src.preprocessing.math_lexicon import calculate_math_density, has_math_content


def compute_math_density(df: pd.DataFrame) -> pd.DataFrame:
    """Add Arabic math density and binary math content columns."""
    df = df.copy()
    df["math_density_arabic"] = df["text_arabic_normalized"].apply(calculate_math_density)
    df["has_math_arabic"] = df["text_arabic_normalized"].apply(has_math_content)
    return df
