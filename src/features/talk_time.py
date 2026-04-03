"""Talk time (word count) analysis per speaker role."""

import pandas as pd

from src.preprocessing.tokenize_ar import count_arabic_words


def compute_talk_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add Arabic word count column to dataframe."""
    df = df.copy()
    df["talktime_words_arabic"] = df["text_arabic"].apply(count_arabic_words)
    return df


def talk_time_by_speaker(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate talk time statistics by speaker role."""
    groups = df.groupby("speaker").agg(
        total_words_en=("talktime_words", "sum"),
        total_words_ar=("talktime_words_arabic", "sum"),
        mean_words_en=("talktime_words", "mean"),
        mean_words_ar=("talktime_words_arabic", "mean"),
        utterance_count=("text", "count"),
    )
    groups["pct_words_en"] = groups["total_words_en"] / groups["total_words_en"].sum()
    groups["pct_words_ar"] = groups["total_words_ar"] / groups["total_words_ar"].sum()
    return groups.reset_index()
