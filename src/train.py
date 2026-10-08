from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import (  # noqa: E402
    build_restaurant_table, clean_metadata, clean_reviews, load_raw, sentiment_label,
)
from models import (  # noqa: E402
    RANDOM_STATE, _fit_algo, build_feature_matrix, evaluate_clustering, extract_topics,
    kmeans_stability, name_clusters, signature_cuisines, score_reviews, train_sentiment,
)

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> dict:
    ART.mkdir(exist_ok=True)
    meta_raw, rev_raw = load_raw()
    meta, reviews = clean_metadata(meta_raw), clean_reviews(rev_raw)
    reviews["Sentiment"] = reviews["Rating"].apply(sentiment_label)
    print(f"Metadata {meta.shape}, reviews {rev_raw.shape} -> {reviews.shape} after cleaning "
          f"({reviews['Sentiment'].notna().sum()} labelled, {reviews['Sentiment'].isna().sum()} neutral)")

    # ---- Sentiment ------------------------------------------------------- #
    sent_model, sent_report = train_sentiment(reviews)
    reviews["Sentiment_Prob"] = score_reviews(reviews, sent_model)
    best_tuned = [r for r in sent_report["comparison"] if r["stage"] == "tuned"][0]
    print("Sentiment (tuned LR) test:", {k: round(v, 3) for k, v in best_tuned.items() if isinstance(v, float)})

    # ---- Restaurant table + clustering ----------------------------------- #
    rest = build_restaurant_table(meta, reviews)
    X, scaler, feat_names = build_feature_matrix(rest)
    grid, best = evaluate_clustering(X)
    k = int(best["k"])
    labels = _fit_algo(best["algorithm"], k, X)
    rest["Cluster"] = labels
    names = name_clusters(rest)
    rest["Cluster_Name"] = rest["Cluster"].map(names)

    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    coords = pca.transform(X)
    rest["PC1"], rest["PC2"] = coords[:, 0], coords[:, 1]
    stability = kmeans_stability(X, k) if best["algorithm"] == "KMeans" else float("nan")
    print(f"Best clustering: {best['algorithm']} k={k} silhouette={best['silhouette']:.3f} "
          f"DB={best['davies_bouldin']:.3f} stability(ARI)={stability:.3f}")

    # ---- Topics ----------------------------------------------------------- #
    topics = {
        "negative": extract_topics(reviews.loc[reviews["Rating"] <= 2, "Clean_Review"]),
        "positive": extract_topics(reviews.loc[reviews["Rating"] >= 4.5, "Clean_Review"]),
    }

    # ---- Persist ----------------------------------------------------------- #
    joblib.dump(sent_model, ART / "sentiment_model.joblib")
    joblib.dump({"scaler": scaler, "feature_names": feat_names, "X": X}, ART / "cluster_features.joblib")
    rest_out = rest.drop(columns=["Cuisine_List"]).assign(Cuisine_List=rest["Cuisine_List"].apply("|".join))
    rest_out.to_csv(ART / "restaurants.csv", index=False)
    reviews.drop(columns=["Review_Words"]).to_csv(ART / "reviews_scored.csv", index=False)
    grid.to_csv(ART / "cluster_grid.csv", index=False)

    summary = {
        "data": {"restaurants_total": int(len(meta)), "restaurants_with_reviews": int(len(rest)),
                 "reviews_raw": int(len(rev_raw)), "reviews_clean": int(len(reviews))},
        "sentiment": sent_report,
        "clustering": {
            "algorithm": best["algorithm"], "k": k,
            "silhouette": float(best["silhouette"]), "davies_bouldin": float(best["davies_bouldin"]),
            "calinski_harabasz": float(best["calinski_harabasz"]),
            "bootstrap_stability_ari": None if np.isnan(stability) else stability,
            "pca_explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
            "cluster_names": {str(a): b for a, b in names.items()},
            "signature_cuisines": {str(a): b for a, b in signature_cuisines(rest).items()},
            "cluster_sizes": {names[int(c)]: int(n) for c, n in rest["Cluster"].value_counts().items()},
        },
        "topics": topics,
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("Artifacts written to", ART)
    return summary


if __name__ == "__main__":
    main()
