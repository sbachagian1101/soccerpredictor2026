# Soccer Prediction Lab 2026

Streamlit app for predicting a forthcoming football match from the last five copied FootyStats match pages for each team.

## Workflow

1. Paste five historical match pages for the upcoming home team.
2. Paste five historical match pages for the upcoming away team.
3. Press **Parse Match Data**.
4. Review the extracted match table and auto-classified match importance.
5. Optionally exclude a match or adjust its importance/opponent-quality multiplier.
6. Press **PREDICT MATCH**.
7. Review 1X2, BTTS, O/U2.5, Corners O/U8.5, the score matrix, team analytics and model breakdown.

## Parser

The parser deliberately uses the actual fixture header, **Final Results** and the first post-match **Data** block. It extracts score, xG, shots, shots on target, possession, corners, cards, fouls and offsides. It does not feed FootyStats' Odds Market, Prediction Stats or Who Will Win sections into the prediction engine.

## Weighting

Each historical observation is weighted by recency, relevant venue, data completeness, competition importance and an editable opponent-quality multiplier. Friendlies are automatically down-weighted.

## Models

The 1X2 ensemble uses 12 methods: results Poisson, xG Poisson, attack/defence Poisson, Dixon-Coles, bivariate Poisson, Negative Binomial, Bayesian Gamma-Poisson, Skellam, Elo-style performance, Power Index, Bayesian form and weighted bootstrap simulation.

BTTS, O/U2.5 and the full-time score matrix come from the ensemble of score-distribution models. Corners O/U8.5 uses a separate five-model ensemble.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment

Use the `main` branch and `streamlit_app.py` as the Streamlit Community Cloud entry point.

Bookmaker odds are not used by the forecasting engine.
