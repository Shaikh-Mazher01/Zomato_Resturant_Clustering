# 🍽️ Zomato — Restaurant Clustering, Sentiment & Streamlit App (Hyderabad)

Segment 100 Hyderabad restaurants on Zomato by price, cuisine, engagement and customer feedback, and classify
review sentiment — delivered as an interactive Streamlit dashboard.

🔗 **Live App:** [View Streamlit Application](https://zomatoresturantclustering-byfahad.streamlit.app/)

## Quick start
```bash
pip install -r requirements.txt
python src/train.py       
streamlit run app.py
pytest -q                  # 11 tests: parsing, text cleaning, every page renders, sentiment prediction
```
Docker: `docker build -t zomato-app . && docker run -p 8501:8501 zomato-app`

## App pages
| Page | What it does |
|---|---|
| Overview | KPIs, cost/cuisine/rating/time charts, key findings |
| Restaurant Explorer | Filter by cost, rating, cuisine, segment, name; CSV export; per-restaurant best/worst reviews |
| Segments | PCA map, segment profiles, signature cuisines, feature heatmap, honest reliability notes |
| Find Similar | "Restaurants like X" via cosine similarity (optional budget guard) |
| Sentiment Lab | Live review scoring with word-level explanation + batch CSV scoring |
| Review Insights | NMF themes behind unhappy/delighted reviews, best/worst restaurants, influential reviewers |
| Model Report | All metrics, model comparison, tuning, confusion matrix, fixes vs. the original notebook |

## Results (reproducible, `random_state=42`)
**Sentiment** (≥4★ vs ≤2★, 20% stratified held-out test; majority baseline 72.1%)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (tuned, GridSearchCV) | 0.952 | 0.980 | 0.953 | 0.966 | 0.988 |
| Calibrated Linear SVM | 0.956 | 0.976 | 0.962 | 0.969 | 0.989 |
| Complement Naive Bayes | 0.945 | 0.956 | 0.968 | 0.962 | 0.986 |

**Clustering** — KMeans, k=3 (chosen over 3 algorithms × k=2–8; silhouette 0.225, Davies-Bouldin 1.62,
bootstrap ARI 0.73):
1. **Budget · Lower-rated** (48 restaurants, median ≈ ₹500) — Biryani / South Indian heavy
2. **Mid-range · Well-rated** (37) — over-represented: Sushi, Café
3. **Premium · Top-rated** (15) — Mediterranean / European, ~3.6 Zomato collections each on average

Silhouette is moderate: with 100 restaurants and price-driven variation the data has overlapping tiers, not
sharply separated groups. The app says so explicitly instead of inflating the numbers.

## What was fixed vs. the original capstone notebook
| Original issue | Fix |
|---|---|
| `Cost` unscaled → PCA PC1 = 99.998% variance, clusters = cost bands | All numerics standardised; cuisine block down-weighted; PCA used only for visualisation |
| Rating `'Like'` wiped the entire row, then filled with 3.5 | Only the rating is invalid; those rows are dropped |
| `"1 Review"` metadata parsed as (0, 0) | Regex parsing; reviewer counts recovered for all rows with no follower info |
| Punctuation removed without spaces ("restaurantfood"); negations deleted | Punctuation → space; `not/no/never` kept |
| TF-IDF fit before split; no seed | Vectoriser inside `Pipeline`; stratified split; seeds |
| Restaurant sentiment features in-sample | Out-of-fold probabilities |
| K-Means claimed (k=6) but code missing; hierarchical used 5 | KMeans, Ward, GMM compared over k=2–8 with silhouette, DB, CH, min-size rule and bootstrap stability |
| LDA scored with silhouette on TF-IDF | NMF themes with readable keywords per sentiment |
| GridSearchCV claimed, not present | Real grid search (n-gram, min_df, C) with 5-fold CV |
| Double `log1p` on a binary column | Removed |
| Conclusion described different work | Findings generated from actual results |
| Colab-only paths, no README/requirements | Relative paths, `requirements.txt`, Dockerfile, tests |

## Limitations
* Sentiment labels derive from star ratings, so the model mostly learns rating language; its real value is scoring
  text that has no rating. Neutral (2.5–3.5★) reviews are excluded from training.
* Only 100 restaurants in one city, 100 reviews each (a sample), so segments describe this dataset, not all of Zomato.
* `Avg_Sentiment` is excluded from clustering because it nearly duplicates `Avg_Rating`.

## Structure
```
notebooks/Zomato_Restaurant_Clustering.ipynb   executed end-to-end analysis notebook (charts + outputs saved)
app.py                Streamlit app
src/data_prep.py      cleaning + feature engineering
src/models.py         sentiment, clustering, topics
src/train.py          end-to-end pipeline → artifacts/
tests/test_app.py     pytest suite
data/                 original CSVs
artifacts/            trained model, tables, metrics (summary.json)
```
