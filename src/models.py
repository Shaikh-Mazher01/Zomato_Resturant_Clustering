from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, calinski_harabasz_score, confusion_matrix,
    davies_bouldin_score, f1_score, precision_score, recall_score, roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from data_prep import NEGATIONS, NUMERIC_FEATURES

RANDOM_STATE = 42
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) - NEGATIONS - {"but", "very", "too", "enough"})


# --------------------------------------------------------------------------- #
# 1. Sentiment classification
# --------------------------------------------------------------------------- #
def _tfidf(**kw) -> TfidfVectorizer:
    return TfidfVectorizer(stop_words=STOP_WORDS, sublinear_tf=True, strip_accents="unicode", **kw)


def _metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def train_sentiment(reviews: pd.DataFrame):
    """Train/compare/tune sentiment models. Vectoriser lives *inside* the pipeline, so there
    is no train/test leakage. Returns (best_pipeline, report_dict)."""
    labeled = reviews.dropna(subset=["Sentiment"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        labeled["Clean_Review"], labeled["Sentiment"].astype(int),
        test_size=0.2, stratify=labeled["Sentiment"], random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": Pipeline([("tfidf", _tfidf(ngram_range=(1, 2), min_df=2)),
                                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Linear SVM (calibrated)": Pipeline([("tfidf", _tfidf(ngram_range=(1, 2), min_df=2)),
                                             ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3))]),
        "Complement Naive Bayes": Pipeline([("tfidf", _tfidf(ngram_range=(1, 2), min_df=2)),
                                            ("clf", ComplementNB())]),
    }
    baseline_acc = float(max(y_te.mean(), 1 - y_te.mean()))

    comparison = []
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        m = _metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])
        cv_f1 = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict")
        m["cv_f1_train"] = f1_score(y_tr, cv_f1)
        comparison.append({"model": name, "stage": "default", **m})

    # Hyper-parameter tuning on the best default family (Logistic Regression)
    grid = GridSearchCV(
        candidates["Logistic Regression"],
        {"tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)], "tfidf__min_df": [1, 2, 3], "clf__C": [0.5, 1, 3, 10]},
        scoring="f1", cv=cv, n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    tuned = _metrics(y_te, best.predict(X_te), best.predict_proba(X_te)[:, 1])
    tuned["cv_f1_train"] = float(grid.best_score_)
    comparison.append({"model": "Logistic Regression", "stage": "tuned", **tuned})

    cm = confusion_matrix(y_te, best.predict(X_te)).tolist()

    # Most influential words
    vec, clf = best.named_steps["tfidf"], best.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    top_neg, top_pos = names[order[:15]].tolist(), names[order[-15:][::-1]].tolist()

    report = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "majority_baseline_accuracy": baseline_acc,
        "comparison": comparison, "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "confusion_matrix": cm, "top_negative_words": top_neg, "top_positive_words": top_pos,
    }

    # Final model for deployment: same tuned configuration, refit on all labelled data
    from sklearn.base import clone
    final = clone(best)
    final.fit(labeled["Clean_Review"], labeled["Sentiment"].astype(int))
    return final, report


def score_reviews(reviews: pd.DataFrame, final_model: Pipeline) -> pd.Series:
    """P(positive) for every review. Labelled reviews get *out-of-fold* probabilities so
    restaurant-level features are not contaminated by in-sample fitting."""
    prob = pd.Series(np.nan, index=reviews.index, dtype=float)
    labeled_idx = reviews.index[reviews["Sentiment"].notna()]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    from sklearn.base import clone
    oof = cross_val_predict(
        clone(final_model), reviews.loc[labeled_idx, "Clean_Review"],
        reviews.loc[labeled_idx, "Sentiment"].astype(int), cv=cv, method="predict_proba",
    )[:, 1]
    prob.loc[labeled_idx] = oof
    rest = reviews.index.difference(labeled_idx)
    if len(rest):
        prob.loc[rest] = final_model.predict_proba(reviews.loc[rest, "Clean_Review"])[:, 1]
    return prob


# --------------------------------------------------------------------------- #
# 2. Clustering
# --------------------------------------------------------------------------- #
def build_feature_matrix(rest: pd.DataFrame):
    """Scale numeric features; down-weight the one-hot cuisine block so the ~20 binary
    columns don't swamp the handful of numeric ones. Returns (X, scaler, feature_names)."""
    cuisine_cols = rest.attrs["cuisine_cols"]
    scaler = StandardScaler().fit(rest[NUMERIC_FEATURES])
    num = scaler.transform(rest[NUMERIC_FEATURES])
    cuis = rest[cuisine_cols].to_numpy(dtype=float)
    cuis = (cuis - cuis.mean(0)) / (cuis.std(0) + 1e-9)
    cuis *= 1.0 / np.sqrt(len(cuisine_cols))  # cuisine block ~ weight of one numeric feature
    X = np.hstack([num, cuis])
    return X, scaler, NUMERIC_FEATURES + cuisine_cols


def _fit_algo(name: str, k: int, X):
    if name == "KMeans":
        return KMeans(k, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
    if name == "Agglomerative (Ward)":
        return AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    return GaussianMixture(k, covariance_type="diag", n_init=5, random_state=RANDOM_STATE).fit(X).predict(X)


def evaluate_clustering(X, k_range=range(2, 9), min_share: float = 0.08):
    """Grid over algorithm x k. A solution is *eligible* if k>=3 and every cluster holds
    at least `min_share` of restaurants (avoids degenerate 'one tiny outlier cluster')."""
    rows = []
    for algo in ["KMeans", "Agglomerative (Ward)", "Gaussian Mixture"]:
        for k in k_range:
            labels = _fit_algo(algo, k, X)
            if len(set(labels)) < 2:
                continue
            smallest = np.bincount(labels).min() / len(labels)
            rows.append({
                "algorithm": algo, "k": k,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "smallest_cluster_share": smallest,
                "eligible": bool(k >= 3 and smallest >= min_share),
            })
    res = pd.DataFrame(rows)
    pool = res[res["eligible"]] if res["eligible"].any() else res
    best = pool.sort_values("silhouette", ascending=False).iloc[0]
    return res, best


def kmeans_stability(X, k: int, n_boot: int = 30) -> float:
    """Mean Adjusted Rand Index between full-data KMeans and KMeans fit on bootstrap samples."""
    rng = np.random.default_rng(RANDOM_STATE)
    base = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(X)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        km = KMeans(k, n_init=10, random_state=int(rng.integers(1e6))).fit(X[idx])
        scores.append(adjusted_rand_score(base.predict(X), km.predict(X)))
    return float(np.mean(scores))


def signature_cuisines(rest: pd.DataFrame, label_col: str = "Cluster", top_n: int = 2, min_count: int = 3) -> dict[int, list[str]]:
    """Cuisines over-represented in a cluster (lift = cluster share / overall share)."""
    overall = rest["Cuisine_List"].explode().value_counts(normalize=True)
    out = {}
    for c, g in rest.groupby(label_col):
        counts = g["Cuisine_List"].explode().value_counts()
        lift = {cu: (n / len(g)) / (overall[cu] * 1.0) for cu, n in counts.items() if n >= min_count}
        ranked = sorted(lift, key=lift.get, reverse=True)[:top_n] or counts.index[:top_n].tolist()
        out[int(c)] = ranked
    return out


def name_clusters(rest: pd.DataFrame, label_col: str = "Cluster") -> dict[int, str]:
    """Human-readable cluster names from price / rating / signature-cuisine profile."""
    cost_q = rest["Cost"].quantile([0.33, 0.66]).values
    rate_med = rest["Avg_Rating"].median()
    sig = signature_cuisines(rest, label_col)
    names, used = {}, set()
    for c, g in rest.groupby(label_col):
        price = "Budget" if g["Cost"].median() <= cost_q[0] else ("Premium" if g["Cost"].median() > cost_q[1] else "Mid-range")
        quality = "Top-rated" if g["Avg_Rating"].mean() >= rate_med + 0.15 else ("Lower-rated" if g["Avg_Rating"].mean() <= rate_med - 0.15 else "Well-rated")
        base = f"{price} · {quality} · {' / '.join(sig[int(c)][:1]) or 'Mixed'}"
        name, i = base, 2
        while name in used:
            name, i = f"{base} ({i})", i + 1
        used.add(name)
        names[int(c)] = name
    return names


# --------------------------------------------------------------------------- #
# 3. Topic modelling (NMF on TF-IDF: more coherent than LDA on short reviews)
# --------------------------------------------------------------------------- #
def extract_topics(texts: pd.Series, n_topics: int = 5, n_words: int = 8) -> list[dict]:
    vec = TfidfVectorizer(stop_words=STOP_WORDS + ["good", "food", "place", "restaurant", "ordered", "order", "but", "very", "just", "really", "like", "also", "even", "would", "got", "get", "one", "time", "enough", "too"],
                          min_df=5, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts)
    nmf = NMF(n_topics, init="nndsvda", random_state=RANDOM_STATE, max_iter=500).fit(X)
    words = np.array(vec.get_feature_names_out())
    share = nmf.transform(X).argmax(1)
    return [
        {"topic": i + 1, "keywords": words[comp.argsort()[-n_words:][::-1]].tolist(),
         "share": float((share == i).mean())}
        for i, comp in enumerate(nmf.components_)
    ]
