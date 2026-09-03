# Soccer Prediction Lab 2026 — Database Edition

A Streamlit soccer prediction application that trains on free historical match data from Football-Data.co.uk and predicts upcoming matches using causal rolling features plus a multi-model ensemble.

## What changed

The original app required five pasted FootyStats match pages for each team. The new database edition can instead:

1. Download historical league CSV files directly from Football-Data.co.uk.
2. Download the current Football-Data fixtures list where available.
3. Build a pre-match training row for every historical fixture using **only information known before kickoff**.
4. Select an upcoming fixture, or choose home/away teams manually.
5. Train and validate multiple statistical/ML models on a chronological holdout.
6. Predict 1X2, BTTS, O/U 2.5, Corners O/U 8.5 and a full correct-score matrix.

## Pre-match features

The feature engine currently includes:

- last-5 and last-10 points per game;
- recent goals scored/conceded and goal difference;
- home-only and away-only form;
- shots, shots on target and corners where available;
- opponent shots/SOT/corners allowed;
- BTTS and O2.5 tendencies;
- recency-weighted form;
- opponent-Elo-adjusted form;
- continuously updated Elo ratings;
- rest days;
- rolling league scoring/draw/BTTS/O2.5/corner baselines;
- attack strength and defensive weakness indices;
- pre-match Poisson scoring intensities;
- home-v-away differential features.

## Model stack

The 1X2 ensemble can use 14 independently generated probability estimates:

1. Form Poisson
2. Dixon-Coles
3. Elo
4. Logistic Regression
5. Random Forest
6. Extra Trees
7. Gradient Boosting
8. Hist Gradient Boosting
9. Gaussian Naive Bayes
10. Poisson goal regression
11. Random-Forest goal model
12. Extra-Trees goal model
13. Gradient goal model
14. Hist-gradient goal model

Models are weighted using chronological validation log loss. BTTS, O/U2.5 and corners also use dedicated binary-model ensembles; BTTS and O/U2.5 are blended with the score-distribution estimates.

## Data leakage protection

Historical rows are generated sequentially. For a match played on date **T**, its features use only matches before **T**. Elo, league averages, form, attack/defence indices and venue splits are all pre-match values.

## Files

- `database_app.py` — database-based Streamlit app (new entry point)
- `data_pipeline.py` — download, cleaning, causal feature engineering and fixture preparation
- `soccer_models.py` — model training, validation, ensemble and score matrix
- `streamlit_app.py` — original paste-based app, retained for compatibility
- `model_core.py`, `parser_core.py`, `prediction_engine.py` — original model/parser code retained

## Run locally

```bash
pip install -r requirements.txt
streamlit run database_app.py
```

## Streamlit Community Cloud

Use:

- branch: `main`
- entry point: `database_app.py`

The app needs outbound internet access to download Football-Data CSVs. If a download is unavailable, use the built-in **Upload CSV** mode.

## Data source

Football-Data.co.uk publishes free downloadable historical results/statistics CSV files and a current fixtures CSV. Bookmaker odds from those files are deliberately **not used as model inputs** in this version.

## Next development layer

The architecture is ready to add xG/xGA, PPDA and richer event data (for example approved/open StatsBomb data or other suitably licensed sources) without changing the basic training/prediction workflow.
