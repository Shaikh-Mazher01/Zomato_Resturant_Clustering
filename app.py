"""Zomato Hyderabad – Restaurant Intelligence Dashboard (Streamlit).

Run:  streamlit run app.py
Artifacts are produced by `python src/train.py` (the app trains them automatically if missing).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Zomato Restaurant Intelligence", page_icon="🍽️", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training models for the first run (≈1–2 min)…")
def ensure_artifacts() -> bool:
    if not (ART / "summary.json").exists():
        from train import main
        main()
    return True


@st.cache_data
def load_tables():
    rest = pd.read_csv(ART / "restaurants.csv")
    rest["Cuisine_List"] = rest["Cuisine_List"].fillna("").str.split("|")
    reviews = pd.read_csv(ART / "reviews_scored.csv", parse_dates=["Time"])
    grid = pd.read_csv(ART / "cluster_grid.csv")
    summary = json.loads((ART / "summary.json").read_text())
    return rest, reviews, grid, summary


@st.cache_resource
def load_sentiment_model():
    return joblib.load(ART / "sentiment_model.joblib")


ensure_artifacts()
rest, reviews, grid, summary = load_tables()
clu = summary["clustering"]
sent = summary["sentiment"]
cluster_order = list(clu["cluster_sizes"].keys())
FEATURE_COLS = ["Cost_Capped", "Total_Cuisine_Count", "Collection_Count", "Opens_Late", "Avg_Rating",
                "Avg_Pictures", "Avg_Reviewer_Followers", "Avg_Review_Words"]

# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🍽️ Zomato Hyderabad")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Restaurant Explorer", "Segments (Clusters)", "Find Similar", "Sentiment Lab",
     "Review Insights", "Model Report"],
)
st.sidebar.caption(f"{summary['data']['restaurants_with_reviews']} restaurants · "
                   f"{summary['data']['reviews_clean']:,} reviews")


def fmt_money(x: float) -> str:
    return f"₹{x:,.0f}"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview():
    st.title("Restaurant Intelligence Dashboard")
    st.write("Segmenting Hyderabad restaurants on Zomato by price, cuisine and customer feedback, "
             "with a sentiment model trained on 10k reviews.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Restaurants", len(rest))
    c2.metric("Reviews analysed", f"{len(reviews):,}")
    c3.metric("Avg rating", f"{reviews['Rating'].mean():.2f} ★")
    c4.metric("Sentiment model F1", f"{sent['comparison'][-1]['f1']:.3f}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(rest, x="Cost", nbins=20, title="Cost for two distribution (₹)",
                           color_discrete_sequence=["#E23744"])
        fig.update_layout(showlegend=False, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cuis = rest["Cuisine_List"].explode().value_counts().head(12).sort_values()
        fig = px.bar(cuis, orientation="h", title="Most common cuisines", color_discrete_sequence=["#E23744"])
        fig.update_layout(showlegend=False, xaxis_title="Restaurants", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        monthly = reviews.assign(Month=reviews["Time"].dt.to_period("M").dt.to_timestamp()) \
            .groupby("Month").agg(Reviews=("Rating", "size"), Rating=("Rating", "mean")).reset_index()
        fig = px.line(monthly, x="Month", y="Reviews", title="Review volume over time",
                      color_discrete_sequence=["#E23744"])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        dist = reviews["Rating"].value_counts().sort_index()
        fig = px.bar(dist, title="Rating distribution", color_discrete_sequence=["#E23744"])
        fig.update_layout(showlegend=False, xaxis_title="Rating", yaxis_title="Reviews")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key findings")
    sizes = clu["cluster_sizes"]
    st.markdown(
        f"- Restaurants fall into **{clu['k']} segments**: "
        + "; ".join(f"**{n}** ({c})" for n, c in sizes.items()) + ".\n"
        f"- Price and rating move together: higher-priced segments earn higher average ratings and a larger "
        f"share of positive reviews (see *Segments*).\n"
        f"- The sentiment model reaches **{sent['comparison'][-1]['f1']:.3f} F1 / "
        f"{sent['comparison'][-1]['roc_auc']:.3f} ROC-AUC** on held-out reviews.\n"
        f"- Complaints centre on taste/quality, service and delivery issues (see *Review Insights*)."
    )


def page_explorer():
    st.title("Restaurant Explorer")
    f1, f2, f3, f4 = st.columns(4)
    cost_lo, cost_hi = int(rest["Cost"].min()), int(rest["Cost"].max())
    cost = f1.slider("Cost for two (₹)", cost_lo, cost_hi, (cost_lo, cost_hi), step=50)
    min_rating = f2.slider("Minimum avg rating", 1.0, 5.0, 1.0, 0.1)
    all_cuisines = sorted(rest["Cuisine_List"].explode().dropna().unique())
    cuisines = f3.multiselect("Cuisines (any of)", all_cuisines)
    segs = f4.multiselect("Segment", cluster_order)
    q = st.text_input("Search by name")

    df = rest[(rest["Cost"].between(*cost)) & (rest["Avg_Rating"] >= min_rating)]
    if cuisines:
        df = df[df["Cuisine_List"].apply(lambda l: any(c in l for c in cuisines))]
    if segs:
        df = df[df["Cluster_Name"].isin(segs)]
    if q:
        df = df[df["Restaurant"].str.contains(q, case=False, na=False)]

    st.caption(f"{len(df)} restaurants match")
    sort_by = st.selectbox("Sort by", ["Avg_Rating", "Cost", "Pct_Positive", "Review_Count"], index=0)
    show = df.sort_values(sort_by, ascending=False)[
        ["Restaurant", "Cost", "Avg_Rating", "Pct_Positive", "Cuisines", "Cluster_Name", "Area", "Links"]
    ]
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config={
            "Cost": st.column_config.NumberColumn("Cost (₹)", format="₹%d"),
            "Avg_Rating": st.column_config.NumberColumn("Avg rating", format="%.2f ★"),
            "Pct_Positive": st.column_config.ProgressColumn("% positive reviews", min_value=0, max_value=1, format="%.0f%%"),
            "Links": st.column_config.LinkColumn("Zomato", display_text="open"),
        },
    )
    st.download_button("Download filtered results (CSV)", show.to_csv(index=False).encode(), "restaurants_filtered.csv")

    if len(df):
        st.divider()
        pick = st.selectbox("Restaurant detail", df["Restaurant"].tolist())
        row = rest[rest["Restaurant"] == pick].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Cost for two", fmt_money(row["Cost"]))
        b.metric("Avg rating", f"{row['Avg_Rating']:.2f} ★")
        c.metric("Positive reviews", f"{row['Pct_Positive']:.0%}")
        st.write(f"**Segment:** {row['Cluster_Name']}  \n**Cuisines:** {row['Cuisines']}  \n**Timings:** {row['Timings']}")
        rv = reviews[reviews["Restaurant"] == pick]
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Most positive reviews**")
            for t in rv.nlargest(3, "Sentiment_Prob")["Review"]:
                st.caption(f"“{str(t)[:220]}…”" if len(str(t)) > 220 else f"“{t}”")
        with cb:
            st.markdown("**Most negative reviews**")
            for t in rv.nsmallest(3, "Sentiment_Prob")["Review"]:
                st.caption(f"“{str(t)[:220]}…”" if len(str(t)) > 220 else f"“{t}”")


def page_segments():
    st.title("Restaurant Segments")
    st.caption(f"{clu['algorithm']} with k={clu['k']}, chosen by silhouette among eligible solutions "
               f"(k≥3, every segment ≥8% of restaurants). 2-D view is a PCA projection "
               f"({clu['pca_explained_variance_2d']:.0%} of variance).")
    fig = px.scatter(rest, x="PC1", y="PC2", color="Cluster_Name", hover_name="Restaurant",
                     hover_data={"Cost": True, "Avg_Rating": ":.2f", "PC1": False, "PC2": False},
                     category_orders={"Cluster_Name": cluster_order})
    fig.update_traces(marker=dict(size=11, line=dict(width=0.5, color="white")))
    fig.update_layout(legend_title_text="Segment", height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment profiles")
    prof = rest.groupby("Cluster_Name").agg(
        Restaurants=("Restaurant", "size"), Median_Cost=("Cost", "median"), Avg_Rating=("Avg_Rating", "mean"),
        Pct_Positive=("Pct_Positive", "mean"), Avg_Cuisines=("Total_Cuisine_Count", "mean"),
        Late_Night_Share=("Opens_Late", "mean"), Avg_Collections=("Collection_Count", "mean"),
    ).reindex(cluster_order).round(2)
    st.dataframe(prof, use_container_width=True)

    sig = clu["signature_cuisines"]
    names = clu["cluster_names"]
    cols = st.columns(len(cluster_order))
    for col, (cid, nm) in zip(cols, names.items()):
        with col:
            st.markdown(f"**{nm}**")
            st.caption("Over-represented cuisines: " + ", ".join(sig[cid]))
            ex = rest[rest["Cluster"] == int(cid)].nlargest(5, "Avg_Rating")["Restaurant"]
            st.caption("Top-rated: " + ", ".join(ex))

    st.subheader("What separates the segments?")
    z = rest.groupby("Cluster_Name")[FEATURE_COLS].mean()
    z = (z - rest[FEATURE_COLS].mean()) / rest[FEATURE_COLS].std()
    fig = px.imshow(z.reindex(cluster_order), color_continuous_scale="RdBu_r", zmin=-2, zmax=2,
                    aspect="auto", labels=dict(color="z-score"))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How reliable are these clusters? (honest assessment)"):
        stab = clu["bootstrap_stability_ari"]
        st.markdown(
            f"- **Silhouette {clu['silhouette']:.2f}**, Davies-Bouldin {clu['davies_bouldin']:.2f}, "
            f"Calinski-Harabasz {clu['calinski_harabasz']:.1f} → moderate, overlapping structure. "
            f"With only {len(rest)} restaurants and mostly price-driven variation, this is the realistic range.\n"
            + (f"- **Bootstrap stability (ARI) {stab:.2f}** → segments are reasonably reproducible.\n" if stab else "")
            + "- Treat segments as *market tiers*, not hard categories: restaurants near a boundary can move.")
        g = grid.copy()
        st.dataframe(g.round(3), use_container_width=True, hide_index=True)


def page_similar():
    st.title("Find Similar Restaurants")
    fv = joblib.load(ART / "cluster_features.joblib")
    X = fv["X"]
    pick = st.selectbox("Pick a restaurant you like", rest["Restaurant"].tolist())
    n = st.slider("How many suggestions", 3, 10, 5)
    same_budget = st.checkbox("Keep within ±30% of its price", value=False)
    i = rest.index[rest["Restaurant"] == pick][0]
    sims = X @ X[i] / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[i]) + 1e-9)
    out = rest.assign(Similarity=sims).drop(index=i)
    if same_budget:
        c0 = rest.loc[i, "Cost"]
        out = out[out["Cost"].between(0.7 * c0, 1.3 * c0)]
    out = out.nlargest(n, "Similarity")[["Restaurant", "Similarity", "Cost", "Avg_Rating", "Cuisines", "Cluster_Name"]]
    st.dataframe(out, hide_index=True, use_container_width=True, column_config={
        "Similarity": st.column_config.ProgressColumn("Similarity", min_value=-1, max_value=1, format="%.2f"),
        "Cost": st.column_config.NumberColumn("Cost (₹)", format="₹%d"),
        "Avg_Rating": st.column_config.NumberColumn("Avg rating", format="%.2f ★"),
    })
    st.caption("Cosine similarity on scaled price, rating, cuisine and engagement features (the same space used for clustering).")


def page_sentiment():
    st.title("Sentiment Lab")
    model = load_sentiment_model()
    examples = {
        "—": "",
        "Positive": "Amazing food and the staff were super friendly. The ambience is great, will definitely visit again!",
        "Negative": "Worst experience ever. The food was cold, the service was slow and the biryani was not fresh.",
        "Mixed": "The taste was good but the delivery was very late and the packaging was poor.",
    }
    ex = st.selectbox("Try an example", list(examples))
    text = st.text_area("Review text", value=examples[ex], height=130, placeholder="Type or paste a restaurant review…")
    if st.button("Analyse", type="primary") and text.strip():
        from data_prep import clean_text
        p = float(model.predict_proba([clean_text(text)])[0, 1])
        label = "Positive 😀" if p >= 0.6 else ("Negative 😞" if p <= 0.4 else "Neutral / mixed 😐")
        a, b = st.columns([1, 2])
        a.metric("Prediction", label)
        a.metric("P(positive)", f"{p:.1%}")
        fig = go.Figure(go.Indicator(mode="gauge", value=p * 100, gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": "#E23744"},
            "steps": [{"range": [0, 40], "color": "#fde2e4"}, {"range": [40, 60], "color": "#f1f1f1"},
                      {"range": [60, 100], "color": "#d8f3dc"}]}))
        fig.update_layout(height=230, margin=dict(t=10, b=0, l=20, r=20))
        b.plotly_chart(fig, use_container_width=True)
        # Word contributions (linear model)
        vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
        Xv = vec.transform([clean_text(text)])
        contrib = Xv.multiply(clf.coef_[0]).tocoo()
        names = vec.get_feature_names_out()
        items = sorted(((names[j], v) for j, v in zip(contrib.col, contrib.data)), key=lambda t: t[1])
        if items:
            df = pd.DataFrame(items, columns=["term", "impact"])
            df["direction"] = np.where(df["impact"] > 0, "pushes positive", "pushes negative")
            fig = px.bar(df, x="impact", y="term", color="direction", orientation="h",
                         color_discrete_map={"pushes positive": "#2d9d5c", "pushes negative": "#E23744"},
                         title="Which words drove this prediction")
            st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Batch scoring")
    up = st.file_uploader("Upload a CSV with a 'Review' column", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "Review" not in df.columns:
            st.error("The CSV needs a column named 'Review'.")
        else:
            from data_prep import clean_text
            df["P_positive"] = model.predict_proba(df["Review"].fillna("").map(clean_text))[:, 1].round(3)
            df["Sentiment"] = np.where(df["P_positive"] >= 0.6, "Positive", np.where(df["P_positive"] <= 0.4, "Negative", "Neutral"))
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", df.to_csv(index=False).encode(), "scored_reviews.csv")


def page_insights():
    st.title("Review Insights")
    t1, t2, t3 = st.tabs(["Themes", "Restaurant sentiment", "Influential reviewers"])
    with t1:
        c1, c2 = st.columns(2)
        for col, key, title in [(c1, "negative", "😞 What unhappy customers talk about (≤2★)"),
                                (c2, "positive", "😀 What delighted customers talk about (≥4.5★)")]:
            with col:
                st.subheader(title)
                for t in summary["topics"][key]:
                    st.markdown(f"**Theme {t['topic']}** · {t['share']:.0%} of reviews  \n`{'` `'.join(t['keywords'][:6])}`")
        st.caption("Themes from NMF topic modelling on TF-IDF (more coherent than LDA for short reviews).")
        wcol1, wcol2 = st.columns(2)
        wcol1.markdown("**Words most associated with negative reviews**")
        wcol1.write(", ".join(sent["top_negative_words"]))
        wcol2.markdown("**Words most associated with positive reviews**")
        wcol2.write(", ".join(sent["top_positive_words"]))
    with t2:
        df = rest.sort_values("Pct_Positive")
        fig = px.bar(df.tail(15), x="Pct_Positive", y="Restaurant", orientation="h", title="Highest share of positive reviews",
                     color_discrete_sequence=["#2d9d5c"])
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.bar(df.head(15), x="Pct_Positive", y="Restaurant", orientation="h", title="Lowest share of positive reviews",
                     color_discrete_sequence=["#E23744"])
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(rest, x="Cost", y="Pct_Positive", color="Cluster_Name", hover_name="Restaurant",
                         title="Does paying more buy happier customers?")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        corr = rest["Cost"].corr(rest["Pct_Positive"], method="spearman")
        st.caption(f"Spearman correlation between cost and % positive reviews: **{corr:.2f}**.")
    with t3:
        rv = reviews.groupby("Reviewer").agg(
            Reviews=("Rating", "size"), Followers=("Reviewer_Followers", "max"),
            Avg_Rating_Given=("Rating", "mean")).reset_index()
        st.markdown("**Most active reviewers in this sample**")
        st.dataframe(rv.nlargest(10, "Reviews").round(2), hide_index=True, use_container_width=True)
        st.markdown("**Reviewers with the largest following**")
        st.dataframe(rv.nlargest(10, "Followers").round(2), hide_index=True, use_container_width=True)


def page_report():
    st.title("Model Report")
    st.subheader("1 · Sentiment classifier")
    st.caption(f"Labels: ≥4★ positive, ≤2★ negative; 2.5–3.5★ treated as neutral and excluded from training. "
               f"Train {sent['n_train']:,} / test {sent['n_test']:,}. Majority-class baseline accuracy: "
               f"{sent['majority_baseline_accuracy']:.1%}. The TF-IDF vectoriser is inside the pipeline (no leakage).")
    comp = pd.DataFrame(sent["comparison"])
    st.dataframe(comp.round(3), hide_index=True, use_container_width=True)
    st.write("**Best hyper-parameters (GridSearchCV, 5-fold, F1):**", sent["best_params"])
    cm = np.array(sent["confusion_matrix"])
    fig = px.imshow(cm, text_auto=True, x=["Pred negative", "Pred positive"], y=["True negative", "True positive"],
                    color_continuous_scale="Reds", title="Confusion matrix (test set)")
    st.plotly_chart(fig, use_container_width=False)
    st.markdown(
        "**Business reading:** *precision* = share of reviews flagged positive that truly are (protects against "
        "promoting unhappy-customer restaurants); *recall* = share of genuinely positive reviews we catch; "
        "for complaint monitoring, watch recall on the **negative** class too.")
    st.warning("Because labels come from star ratings, this model mostly learns *how people phrase ratings*. "
               "Its value is classifying text **without** a star rating (comments, social posts, support tickets).")

    st.subheader("2 · Clustering")
    st.write(f"**{clu['algorithm']}, k={clu['k']}** — silhouette {clu['silhouette']:.3f}, "
             f"Davies-Bouldin {clu['davies_bouldin']:.3f}, Calinski-Harabasz {clu['calinski_harabasz']:.1f}"
             + (f", bootstrap ARI {clu['bootstrap_stability_ari']:.2f}" if clu["bootstrap_stability_ari"] else ""))
    fig = px.line(grid, x="k", y="silhouette", color="algorithm", markers=True, title="Silhouette by algorithm and k")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Features are standardised (the original notebook left Cost unscaled, so PCA's first component "
               "captured 99.998% of variance and the clusters were just cost bands).")

    st.subheader("3 · Data quality fixes vs. the original notebook")
    st.markdown(
        "- 'Like' ratings no longer wipe whole rows or get imputed with 3.5.\n"
        "- Reviewer metadata parsed correctly (`1 Review` keeps its count).\n"
        "- Punctuation replaced by spaces; negations preserved.\n"
        "- Vectoriser fit only on training folds; out-of-fold probabilities feed restaurant features.")


{"Overview": page_overview, "Restaurant Explorer": page_explorer, "Segments (Clusters)": page_segments,
 "Find Similar": page_similar, "Sentiment Lab": page_sentiment, "Review Insights": page_insights,
 "Model Report": page_report}[page]()
