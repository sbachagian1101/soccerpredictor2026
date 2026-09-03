from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_pipeline import (
    FEATURE_COLUMNS, LEAGUES, available_teams, build_feature_dataset,
    build_prediction_row, default_seasons, download_fixtures,
    load_football_data, load_uploaded_csvs, recent_team_matches, season_label,
)
from soccer_models import train_and_predict

st.set_page_config(page_title="Soccer Prediction Lab 2026", page_icon="⚽", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.15); padding: 12px; border-radius: 12px;}
.hero {padding: 1.0rem 1.2rem; border: 1px solid rgba(127,127,127,.18); border-radius: 16px; margin-bottom: 1rem;}
.subtle {opacity: .72; font-size: .92rem;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero"><h1 style="margin-bottom:.25rem">⚽ Soccer Prediction Lab 2026</h1>
<div class="subtle">Historical database → causal features → multi-model ensemble → upcoming match prediction</div></div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_download(league: str, seasons: tuple[int, ...]):
    return load_football_data([league], seasons)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_fixtures():
    return download_fixtures()

@st.cache_data(show_spinner=False)
def cached_features(raw: pd.DataFrame, min_history: int):
    return build_feature_dataset(raw, min_history=min_history)

def pct(x: float) -> str:
    return f"{100*x:.1f}%"

def fmt_value(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v): return "—"
        return round(float(v), 3)
    return v

for key, value in {"raw_data": None, "download_errors": [], "fixtures": None, "prediction": None, "active_league": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("Data & training")
    mode = st.radio("Data source", ["Automatic Football-Data", "Upload CSV"], index=0)
    league = st.selectbox("Competition", list(LEAGUES), format_func=lambda x: LEAGUES[x])
    min_history = st.slider("Minimum prior matches/team", 3, 10, 5, 1,
        help="Training rows are created only after each team has this many earlier matches.")

    if mode == "Automatic Football-Data":
        season_options = list(range(default_seasons(10)[0], default_seasons(1)[0] + 1))
        seasons = st.multiselect("Training seasons", season_options, default=default_seasons(5), format_func=season_label,
            help="Five seasons is a good initial balance between recency and sample size.")
        if st.button("⬇️ Load / refresh data", type="primary", use_container_width=True):
            if not seasons:
                st.error("Select at least one season.")
            else:
                with st.spinner("Downloading historical match files and latest fixtures…"):
                    try:
                        raw, errors = cached_download(league, tuple(sorted(seasons)))
                        st.session_state.raw_data = raw
                        st.session_state.download_errors = errors
                        st.session_state.active_league = league
                        st.session_state.prediction = None
                        try:
                            st.session_state.fixtures = cached_fixtures()
                        except Exception as exc:
                            st.session_state.fixtures = None
                            st.session_state.download_errors = errors + [f"Fixtures: {exc}"]
                        st.success(f"Loaded {len(raw):,} match rows.")
                    except Exception as exc:
                        st.error(str(exc))
    else:
        files = st.file_uploader("Football-Data-style CSV files", type=["csv"], accept_multiple_files=True)
        if st.button("Load uploaded CSVs", type="primary", use_container_width=True):
            if not files:
                st.error("Upload one or more CSV files first.")
            else:
                try:
                    raw = load_uploaded_csvs(files)
                    st.session_state.raw_data = raw
                    st.session_state.download_errors = []
                    codes = raw["league_code"].astype(str).unique().tolist()
                    st.session_state.active_league = league if league in codes else str(raw["league_code"].iloc[-1])
                    st.session_state.prediction = None
                    st.success(f"Loaded {len(raw):,} rows.")
                except Exception as exc:
                    st.error(str(exc))
    st.divider()
    st.caption("Prediction features never include bookmaker odds. Odds may be retained later only for independent market benchmarking.")

raw = st.session_state.raw_data
if raw is None:
    st.info("Use the sidebar to load historical data. For the first run, choose a league and the latest five seasons, then click **Load / refresh data**.")
    st.markdown("### Version 1 model stack")
    st.write("Form Poisson, Dixon-Coles, Elo, Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, Gaussian Naive Bayes, Poisson goal regression, Random-Forest goal model, Extra-Trees goal model, Gradient goal model and Hist-gradient goal model.")
    st.stop()

if st.session_state.download_errors:
    with st.expander(f"Data source notices ({len(st.session_state.download_errors)})"):
        for e in st.session_state.download_errors:
            st.write("•", e)

loaded_leagues = sorted(raw["league_code"].dropna().astype(str).unique())
active_league = league if league in loaded_leagues else st.session_state.active_league
if active_league not in loaded_leagues:
    active_league = loaded_leagues[0]
league_raw = raw[raw["league_code"].astype(str) == str(active_league)].copy()
training_features = cached_features(league_raw, min_history)

summary_cols = st.columns(4)
summary_cols[0].metric("Raw matches", f"{len(league_raw):,}")
summary_cols[1].metric("Training rows", f"{len(training_features):,}")
summary_cols[2].metric("Teams", f"{len(available_teams(league_raw, active_league))}")
summary_cols[3].metric("Latest result", league_raw["Date"].max().strftime("%d %b %Y") if len(league_raw) else "—")

predict_tab, data_tab, validation_tab, inspector_tab = st.tabs(["🎯 Predict match", "🧠 Training data", "📊 Model validation", "🔎 Data inspector"])

with predict_tab:
    st.subheader("Select the match")
    fixtures = st.session_state.fixtures
    fixture_rows = pd.DataFrame()
    if fixtures is not None and not fixtures.empty and "Div" in fixtures.columns:
        fixture_rows = fixtures[fixtures["Div"].astype(str) == str(active_league)].copy()
    source_choice = "Manual teams"
    if not fixture_rows.empty:
        source_choice = st.radio("Match selection", ["Latest fixture list", "Manual teams"], horizontal=True)
    teams = available_teams(league_raw, active_league)
    fixture_date = pd.Timestamp(date.today() + timedelta(days=1))
    if source_choice == "Latest fixture list":
        fixture_rows = fixture_rows.reset_index(drop=True)
        labels = [f"{r.Date.strftime('%d %b %Y')} · {r.HomeTeam} vs {r.AwayTeam}" for _, r in fixture_rows.iterrows()]
        ix = st.selectbox("Upcoming fixture", range(len(labels)), format_func=lambda i: labels[i])
        fr = fixture_rows.iloc[int(ix)]
        home_team, away_team, fixture_date = str(fr["HomeTeam"]), str(fr["AwayTeam"]), pd.Timestamp(fr["Date"])
        if home_team not in teams or away_team not in teams:
            st.warning("One team has little/no history in the loaded seasons. Add an earlier season or choose manual teams if needed.")
    else:
        c1, c2, c3 = st.columns([1, 1, .7])
        home_team = c1.selectbox("Home team", teams, index=0)
        away_options = [t for t in teams if t != home_team]
        away_team = c2.selectbox("Away team", away_options, index=min(1, len(away_options)-1))
        fixture_date = pd.Timestamp(c3.date_input("Prediction date", value=date.today() + timedelta(days=1)))

    if fixture_date <= league_raw["Date"].max():
        st.caption("Backtest mode: only matches before the selected prediction date will be used for training and features.")

    if st.button("⚡ PREDICT MATCH", type="primary", use_container_width=True):
        with st.spinner("Building pre-match features, training models and running the ensemble…"):
            try:
                cutoff_features = training_features[training_features["Date"] < fixture_date].copy()
                pred_row = build_prediction_row(league_raw, active_league, home_team, away_team, as_of=fixture_date)
                st.session_state.prediction = train_and_predict(cutoff_features, pred_row)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    bundle = st.session_state.prediction
    if bundle is not None:
        st.divider()
        st.markdown(f"## {bundle.home_team} vs {bundle.away_team}")
        p = bundle.probabilities
        winner = max([("Home", p["home"]), ("Draw", p["draw"]), ("Away", p["away"])], key=lambda x: x[1])
        st.caption(f"Highest 1X2 probability: **{winner[0]} ({pct(winner[1])})** · Ensemble expected goals {bundle.expected_goals[0]:.2f}–{bundle.expected_goals[1]:.2f}")
        a, b, c, d, e = st.columns(5)
        a.metric("1 · Home", pct(p["home"])); b.metric("X · Draw", pct(p["draw"])); c.metric("2 · Away", pct(p["away"])); d.metric("BTTS · Yes", pct(p["btts_yes"])); e.metric("Over 2.5", pct(p["over25"]))
        c1, c2, c3 = st.columns(3)
        c1.metric(f"xG · {bundle.home_team}", f"{bundle.expected_goals[0]:.2f}")
        c2.metric(f"xG · {bundle.away_team}", f"{bundle.expected_goals[1]:.2f}")
        c3.metric("Corners O8.5", pct(bundle.corners_probability) if bundle.corners_probability is not None else "N/A")

        left, right = st.columns([1.1, .9])
        with left:
            st.markdown("### Correct-score probability matrix")
            labels = [str(i) for i in range(bundle.score_matrix.shape[0])]
            fig = px.imshow(bundle.score_matrix * 100, x=labels, y=labels, text_auto=".1f",
                labels={"x": f"{bundle.away_team} goals", "y": f"{bundle.home_team} goals", "color": "%"}, aspect="auto")
            fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("### Most likely scores")
            top_df = pd.DataFrame(bundle.top_scores[:8], columns=["Score", "Probability"])
            top_df["Probability"] = top_df["Probability"].map(lambda x: f"{100*x:.2f}%")
            st.dataframe(top_df, hide_index=True, use_container_width=True)
            st.markdown("### Model consensus")
            consensus = bundle.method_table.set_index("Method")[["Home %", "Draw %", "Away %"]]
            st.bar_chart(consensus, horizontal=True, height=330)

        st.markdown("### Individual 1X2 methods")
        mt = bundle.method_table.copy()
        for col in ["Home %", "Draw %", "Away %", "Ensemble weight"]:
            mt[col] = mt[col].map(lambda x: round(float(x), 2))
        st.dataframe(mt, hide_index=True, use_container_width=True)

        with st.expander("Why the model produced this prediction"):
            r = bundle.feature_snapshot.set_index("Feature")["Value"]
            explain = pd.DataFrame({
                "Indicator": ["L5 points/game", "L10 points/game", "Venue L10 points/game", "L10 goal difference/game", "L5 shots on target", "Elo", "Attack strength", "Defensive weakness", "Rest days"],
                bundle.home_team: [r.get("home_ppg_l5"), r.get("home_ppg_l10"), r.get("home_venue_ppg_l10"), r.get("home_gd_l10"), r.get("home_sot_l5"), r.get("elo_home"), r.get("home_attack_strength"), r.get("home_def_weakness"), r.get("home_rest_days")],
                bundle.away_team: [r.get("away_ppg_l5"), r.get("away_ppg_l10"), r.get("away_venue_ppg_l10"), r.get("away_gd_l10"), r.get("away_sot_l5"), r.get("elo_away"), r.get("away_attack_strength"), r.get("away_def_weakness"), r.get("away_rest_days")],
            })
            st.dataframe(explain.map(fmt_value), hide_index=True, use_container_width=True)
            for note in bundle.data_notes: st.caption("• " + note)

with data_tab:
    st.subheader("Causally engineered training table")
    st.write("Each row represents one historical match. Every predictor in that row was calculated from matches available **before kickoff**. The actual result columns are retained only as training targets.")
    if training_features.empty:
        st.warning("There are not yet enough prior matches to create training rows with the selected minimum-history setting.")
    else:
        display_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "result", "btts", "over25", "total_corners"] + FEATURE_COLUMNS
        st.dataframe(training_features[[c for c in display_cols if c in training_features.columns]].tail(500), use_container_width=True, height=520)
        st.download_button("Download engineered training CSV", training_features.to_csv(index=False).encode("utf-8"),
            file_name=f"{active_league}_soccer_training_features.csv", mime="text/csv")

with validation_tab:
    st.subheader("Chronological validation")
    bundle = st.session_state.prediction
    if bundle is None:
        st.info("Run a prediction first. The app will then show each model's performance on the most recent chronological holdout sample.")
    else:
        vt = bundle.validation_table.copy()
        vt["Validation accuracy"] = vt["Validation accuracy"].map(lambda x: round(100*x, 2) if pd.notna(x) else np.nan)
        vt["Log loss"] = vt["Log loss"].map(lambda x: round(float(x), 4) if pd.notna(x) else np.nan)
        st.dataframe(vt, hide_index=True, use_container_width=True)
        valid = vt[vt["Log loss"].notna()]
        if not valid.empty:
            best = valid.sort_values("Log loss").iloc[0]
            st.success(f"Best calibrated validation method on this run: **{best['Model']}** (log loss {best['Log loss']}).")
        st.caption("Validation is time-ordered. Later matches are held out; the model is not allowed to train on them first.")

with inspector_tab:
    st.subheader("Raw match data & recent team form")
    c1, c2 = st.columns([1, 1])
    inspect_team = c1.selectbox("Team to inspect", available_teams(league_raw, active_league), key="inspect_team")
    n_recent = c2.slider("Recent matches", 5, 20, 10)
    st.dataframe(recent_team_matches(league_raw, active_league, inspect_team, n_recent), hide_index=True, use_container_width=True)
    with st.expander("Raw source rows"):
        st.dataframe(league_raw.sort_values("Date", ascending=False).head(300), use_container_width=True, height=450)

st.divider()
st.caption("Model outputs are probabilistic estimates, not guarantees. Version 1 is intentionally league-specific and excludes bookmaker odds from model inputs. The next enrichment layer can add xG/xGA, PPDA and event data where licensing and coverage are suitable.")
