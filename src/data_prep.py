from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Negations carry sentiment, so they must survive stop-word removal.
NEGATIONS = {"no", "not", "nor", "never", "n't", "nothing", "nobody", "neither", "without"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(data_dir / "metadata.csv")
    reviews = pd.read_csv(data_dir / "reviews.csv")
    return meta, reviews


# --------------------------------------------------------------------------- #
# Metadata cleaning
# --------------------------------------------------------------------------- #
def _count_collections(value) -> int:
    return 0 if pd.isna(value) else len([c for c in str(value).split(",") if c.strip()])


def _opens_late(timing: str) -> int:
    """1 if the restaurant stays open past midnight (e.g. '12 Noon to 2 AM', 'to 12midnight')."""
    if not isinstance(timing, str):
        return 0
    t = timing.lower()
    return int(bool(re.search(r"(12\s*midnight|\b[1-5]\s*am\b|\b[1-5]:\d\d\s*am\b)", t)))


def clean_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    df = meta.rename(columns={"Name": "Restaurant"}).copy()
    df["Restaurant"] = df["Restaurant"].str.strip()
    df["Cost"] = df["Cost"].astype(str).str.replace(",", "", regex=False).astype(int)
    df["Cuisines"] = df["Cuisines"].fillna("").astype(str)
    df["Cuisine_List"] = df["Cuisines"].apply(lambda s: [c.strip() for c in s.split(",") if c.strip()])
    df["Total_Cuisine_Count"] = df["Cuisine_List"].str.len()
    df["Collection_Count"] = df["Collections"].apply(_count_collections)
    df["Timings"] = df["Timings"].fillna("Not available")
    df["Opens_Late"] = df["Timings"].apply(_opens_late)
    # Area slug from the URL, e.g. .../hyderabad/paradise-gachibowli -> 'gachibowli'
    df["Area"] = df["Links"].str.rstrip("/").str.split("/").str[-1].str.split("-").str[-1].str.title()
    return df


# --------------------------------------------------------------------------- #
# Review cleaning
# --------------------------------------------------------------------------- #
_REV_RE = re.compile(r"(\d+)\s*Review", re.I)
_FOL_RE = re.compile(r"(\d+)\s*Follower", re.I)


def parse_reviewer_metadata(s) -> tuple[float, float]:
    """'96 Reviews , 1069 Followers' -> (96, 1069); '1 Review' -> (1, 0)."""
    if not isinstance(s, str):
        return (np.nan, np.nan)
    rev = _REV_RE.search(s)
    fol = _FOL_RE.search(s)
    return (float(rev.group(1)) if rev else 0.0, float(fol.group(1)) if fol else 0.0)


def clean_text(text: str) -> str:
    """Lower-case, strip URLs/digits, replace punctuation with spaces (never glue words)."""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"'", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.drop_duplicates().copy()
    df["Restaurant"] = df["Restaurant"].str.strip()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")  # 'Like' -> NaN
    df = df.dropna(subset=["Restaurant", "Review", "Rating", "Time"]).copy()
    parsed = df["Metadata"].apply(parse_reviewer_metadata)
    df["Reviewer_Reviews"] = parsed.str[0]
    df["Reviewer_Followers"] = parsed.str[1]
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])
    df["Year"] = df["Time"].dt.year
    df["Month"] = df["Time"].dt.month
    df["Clean_Review"] = df["Review"].apply(clean_text)
    df["Review_Words"] = df["Clean_Review"].str.split().str.len()
    df = df[df["Review_Words"] > 0].reset_index(drop=True)
    return df.drop(columns=["Metadata"])


def sentiment_label(rating: float) -> float:
    """Ratings >=4 positive, <=2 negative, 2.5-3.5 neutral (NaN -> excluded from training)."""
    if rating >= 4:
        return 1.0
    if rating <= 2:
        return 0.0
    return np.nan


# --------------------------------------------------------------------------- #
# Restaurant-level feature table (the input to clustering)
# --------------------------------------------------------------------------- #
def cap_iqr(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def build_restaurant_table(meta: pd.DataFrame, reviews: pd.DataFrame, min_cuisine_freq: int = 3) -> pd.DataFrame:
    """One row per restaurant that has reviews, with numeric + one-hot cuisine features.

    `reviews` must already contain a `Sentiment_Prob` column (P(positive)).
    """
    agg = reviews.groupby("Restaurant").agg(
        Avg_Rating=("Rating", "mean"),
        Review_Count=("Rating", "size"),
        Avg_Pictures=("Pictures", "mean"),
        Avg_Reviewer_Followers=("Reviewer_Followers", lambda s: cap_iqr(s).mean()),
        Avg_Sentiment=("Sentiment_Prob", "mean"),
        Pct_Positive=("Sentiment_Prob", lambda s: (s >= 0.5).mean()),
        Avg_Review_Words=("Review_Words", "mean"),
    ).reset_index()

    df = meta.merge(agg, on="Restaurant", how="inner")
    df["Cost_Capped"] = cap_iqr(df["Cost"])

    freq = df["Cuisine_List"].explode().value_counts()
    keep = freq[freq >= min_cuisine_freq].index.tolist()
    for c in keep:
        df[f"Cuisine_{c}"] = df["Cuisine_List"].apply(lambda lst, c=c: int(c in lst))
    df.attrs["cuisine_cols"] = [f"Cuisine_{c}" for c in keep]
    return df


NUMERIC_FEATURES = [
    "Cost_Capped", "Total_Cuisine_Count", "Collection_Count", "Opens_Late",
    "Avg_Rating", "Avg_Pictures", "Avg_Reviewer_Followers", "Avg_Review_Words",
]
# NOTE: The sentiment model is trained on star ratings, so they are near-duplicates of Avg_Rating.
