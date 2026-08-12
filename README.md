# Soccer Predictor v2

Generic multi-model 1X2 soccer prediction app for Windows and Streamlit/mobile web.

## What is new in v2
- Match-specific **Explanation** tab explaining why the model prefers 1 / X / 2.
- Separate **Attack Index** and **Defence Index** (100 = comparison-group average).
- Separate **Attacking Strength**, **Attacking Weakness**, **Defensive Strength**, and **Defensive Weakness** based on favourable/unfavourable component deviations.
- Component-level attack and defensive-vulnerability ratings for goals, xG and shots on target.
- Missing optional xG/SOT metrics are excluded and remaining weights are re-normalised.
- Default ensemble adjusted to 60% Poisson/Dixon-Coles, 20% Elo, 15% recent form, 5% attack-defence after a preliminary walk-forward check on the supplied Finland data. These weights are still defaults, not universal calibrated weights.

## Windows
1. Install Python 3.11+ if Python is not already installed.
2. Unzip this folder.
3. Double-click `run.bat`.
4. Click **Upload CSV file(s)** and select one or multiple season/history files.
5. Select **Home team** and **Away team** from the dropdowns.
6. Click **ANALYSE MATCH**.

If the uploaded files already contain upcoming fixtures, the optional fixture dropdown can auto-fill the home and away selections and preserve the exact kickoff cutoff.

## Streamlit / mobile
Run locally with `run_streamlit.bat`, or upload the project to GitHub and deploy `streamlit_app.py` on Streamlit Community Cloud.

### GitHub + Streamlit Cloud
1. Create a new GitHub repository.
2. Upload `model_core.py`, `streamlit_app.py`, and `requirements.txt`.
3. Open Streamlit Community Cloud and create a new app from the repository.
4. Set the main file to `streamlit_app.py`.
5. Deploy. The resulting URL works on desktop and mobile browsers and can be added to a phone home screen.

## CSV fields
The app supports the supplied FootyStats-style files directly and also maps several common aliases.

Minimum information:
- date/timestamp
- match status (or goals sufficient to infer completed rows)
- home team
- away team
- full-time home goals
- full-time away goals

Optional but useful:
- home/away xG
- home/away shots
- home/away shots on target

Common aliases supported include `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `HS`, `AS`, `HST`, and `AST`.

## Attack / defence profile
Attack is built from 55% goals scored, 30% xG and 15% shots on target. Defensive vulnerability is built from 55% goals conceded, 30% xGA and 15% opponent shots on target. If an optional component is absent, its weight is redistributed across available components.

Each metric combines:
- 30% available-history/season overall
- 25% relevant home/away split
- 25% last 5 overall
- 20% last 5 at the relevant venue

Small samples are shrunk toward the comparison-group average.

The app reports:
- **Attack Index** — 100 = average; higher is better.
- **Defence Index** — 100 = average; higher is better.
- **Attacking Strength / Weakness** — weighted positive and negative deviations of the attack components from average. A team can have both simultaneously.
- **Defensive Strength / Weakness** — weighted positive and negative deviations of the defensive components from average. A team can have both simultaneously.

## Prediction models
Final 1X2 default ensemble:
- 60% Poisson / Dixon-Coles
- 20% Elo
- 15% recent-form model
- 5% attack-v-defence logistic model

The attack-defence model is still displayed separately in Model Breakdown even though it has a low ensemble weight.

## Validation status
The design is a strong analytical prototype, but it is not yet a fully calibrated forecasting system. The feature and recency weights remain fixed rules. For production-quality probabilities, the app should next add automated walk-forward backtesting and optimise/calibrate weights per competition using log loss, Brier score and calibration curves.

Bookmaker odds are not used.

## Leakage protection
For an upcoming fixture present in the CSV, the engine excludes all matches at or after that kickoff. Manual pairings use the latest completed data available.
