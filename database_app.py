from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from footystats_parser import (
    away_team_options,
    build_venue_team_data,
    completed_matches,
    home_team_options,
    load_footystats_uploads,
    venue_summary,
)
from prediction_engine import build_prediction, feature_comparison, score_matrix_dataframe

st.set_page_config(page_title="Soccer Prediction Lab 2026", page_icon="⚽", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.15); padding: 12px; border-radius: 12px;}
.hero {padding: 1.05rem 1.25rem; border: 1px solid rgba(127,127,127,.18); border-radius: 16px; margin-bottom: 1rem;}
.subtle {opacity: .74; font-size: .93rem;}
.venue-note {padding: .8rem 1rem; border-radius: 12px; background: rgba(127,127,127,.06); border: 1px solid rgba(127,127,127,.15);}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
<h1 style="margin-bottom:.25rem">⚽ Soccer Prediction Lab 2026</h1>
<div class="subtle">FootyStats uploads → home-only vs away-only team data → 12-method mathematical ensemble</div>
</div>
""",
    unsafe_allow_html=True,
)


def pct(v: float) -> str:
    return f"{100 * float(v):.1f}%"


def fmt(v, digits=2):
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "—"
    return f"{float(v):.{digits}f}"


for key, default in {
    "footystats_raw": None,
    "upload_notices": [],
    "prediction": None,
    "prediction_key": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("FootyStats data")
    files = st.file_uploader(
        "Upload one or more FootyStats match CSVs",
        type=["csv"],
        accept_multiple_files=True,
        help="You can upload several seasons of the same competition. Duplicate matches are removed automatically.",
    )
    if st.button("📂 Parse uploaded datasets", type="primary", use_container_width=True):
        if not files:
            st.error("Upload at least one FootyStats CSV first.")
        else:
            try:
                raw, notices = load_footystats_uploads(files)
                st.session_state.footystats_raw = raw
                st.session_state.upload_notices = notices
                st.session_state.prediction = None
                st.session_state.prediction_key = None
                st.success(f"Parsed {len(raw):,} unique match rows.")
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.markdown("**Prediction rule**")
    st.caption(
        "Home selection uses only completed matches where that team was the HOME team. "
        "Away selection uses only completed matches where that team was the AWAY team."
    )
    st.caption("Future/incomplete fixtures and bookmaker odds are excluded from the prediction engine.")

raw = st.session_state.footystats_raw
if raw is None:
    st.info(
        "Upload your FootyStats season CSV files in the sidebar and click **Parse uploaded datasets**. "
        "The app will then create separate Home Team and Away Team dropdowns."
    )
    st.markdown("### Mathematical methods retained")
    st.write(
        "Poisson (results), xG Poisson, Attack–Defence Poisson, Dixon–Coles, Bivariate Poisson, "
        "Negative Binomial, Bayesian Gamma–Poisson, Skellam Difference, Elo-style Performance, "
        "Power Index, Bayesian Form and Weighted Bootstrap simulation."
    )
    st.stop()

completed = completed_matches(raw)
future_count = int((~raw["is_completed"]).sum())
home_teams = home_team_options(raw)
away_teams = away_team_options(raw)

if not home_teams or not away_teams:
    st.error("The uploaded files do not contain enough completed home/away matches to create team dropdowns.")
    st.stop()

if st.session_state.upload_notices:
    with st.expander(f"Upload notices ({len(st.session_state.upload_notices)})"):
        for notice in st.session_state.upload_notices:
            st.write("•", notice)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Uploaded rows", f"{len(raw):,}")
m2.metric("Completed matches used", f"{len(completed):,}")
m3.metric("Future/non-complete excluded", f"{future_count:,}")
m4.metric("Teams available", f"{len(set(home_teams) | set(away_teams)):,}")

st.markdown("## Select the match")
c1, c2 = st.columns(2)
home_team = c1.selectbox("🏠 Home team", home_teams, key="selected_home")
valid_away = [t for t in away_teams if t != home_team]
away_team = c2.selectbox("✈️ Away team", valid_away, key="selected_away")

home_df = build_venue_team_data(raw, home_team, "H")
away_df = build_venue_team_data(raw, away_team, "A")
hsum, asum = venue_summary(home_df), venue_summary(away_df)

st.markdown(
    f"""
<div class="venue-note">
<b>Data entering this prediction:</b> {home_team}: <b>{len(home_df)} home matches only</b> ·
{away_team}: <b>{len(away_df)} away matches only</b>. No home-team away matches and no away-team home matches are included.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Venue-specific sample")
a, b, c, d, e, f = st.columns(6)
a.metric(f"{home_team} home matches", len(home_df))
b.metric("Home PPG", fmt(hsum.get("ppg")))
c.metric("Home xG", fmt(hsum.get("xg")))
d.metric(f"{away_team} away matches", len(away_df))
e.metric("Away PPG", fmt(asum.get("ppg")))
f.metric("Away xG", fmt(asum.get("xg")))

if len(home_df) < 3 or len(away_df) < 3:
    st.warning("One selected team has fewer than 3 completed venue-specific matches. The model can run, but the sample is very small.")

if st.button("⚡ PREDICT MATCH", type="primary", use_container_width=True):
    try:
        if home_df.empty or away_df.empty:
            raise ValueError("Both teams need at least one completed match at the selected venue.")
        with st.spinner("Running the venue-specific mathematical ensemble…"):
            result = build_prediction(home_df, away_df)
        st.session_state.prediction = result
        st.session_state.prediction_key = (home_team, away_team, len(home_df), len(away_df))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

result = st.session_state.prediction
current_key = (home_team, away_team, len(home_df), len(away_df))
if result is not None and st.session_state.prediction_key != current_key:
    st.info("Team selection changed. Click **PREDICT MATCH** to calculate a new prediction from the newly selected venue data.")
    result = None

predict_tab, parsed_tab, analytics_tab, models_tab = st.tabs(
    ["🎯 Prediction", "📋 Parsed venue data", "📊 Team analytics", "🧮 Model breakdown"]
)

with predict_tab:
    if result is None:
        st.info("Click **PREDICT MATCH** to run the 12-method ensemble using the venue-specific data shown above.")
    else:
        st.markdown(f"## {result['home_team']} vs {result['away_team']}")
        pvals = [result["p_home"], result["p_draw"], result["p_away"]]
        labels = [result["home_team"], "Draw", result["away_team"]]
        best = int(np.argmax(pvals))
        st.caption(
            f"Highest 1X2 probability: **{labels[best]} ({pct(pvals[best])})** · "
            f"Ensemble expected goals {result['expected_home_goals']:.2f}–{result['expected_away_goals']:.2f} · "
            f"Model confidence {result['confidence']:.0f}/100"
        )

        a, b, c, d, e = st.columns(5)
        a.metric("1 · Home", pct(result["p_home"]))
        b.metric("X · Draw", pct(result["p_draw"]))
        c.metric("2 · Away", pct(result["p_away"]))
        d.metric("BTTS · Yes", pct(result["btts_yes"]))
        e.metric("Over 2.5", pct(result["over25"]))

        a, b, c = st.columns(3)
        a.metric(f"Expected goals · {result['home_team']}", fmt(result["expected_home_goals"]))
        b.metric(f"Expected goals · {result['away_team']}", fmt(result["expected_away_goals"]))
        c.metric("Corners O8.5", pct(result["corners"]["over85"]))

        left, right = st.columns([1.12, .88])
        with left:
            st.markdown("### Correct-score probability matrix")
            matrix_df = score_matrix_dataframe(result, max_display=6)
            fig = px.imshow(
                matrix_df.values,
                x=matrix_df.columns,
                y=matrix_df.index,
                text_auto=".1f",
                labels={"x": f"{result['away_team']} goals", "y": f"{result['home_team']} goals", "color": "%"},
                aspect="auto",
            )
            fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("### Most likely scores")
            top = pd.DataFrame(result["top_scores"], columns=["Score", "Probability"])
            top["Probability"] = top["Probability"].map(pct)
            st.dataframe(top, hide_index=True, use_container_width=True)
            st.markdown("### Goal markets")
            market = pd.DataFrame({
                "Market": ["BTTS Yes", "BTTS No", "Over 2.5", "Under 2.5", "Corners Over 8.5", "Corners Under 8.5"],
                "Probability": [
                    result["btts_yes"], result["btts_no"], result["over25"], result["under25"],
                    result["corners"]["over85"], result["corners"]["under85"],
                ],
            })
            market["Probability"] = market["Probability"].map(pct)
            st.dataframe(market, hide_index=True, use_container_width=True)

with parsed_tab:
    st.subheader("Exactly which matches are used")
    st.caption(
        "These tables are the direct input observations for the mathematical engine. "
        "Every row below has weight 1.0, so all completed uploaded venue-specific matches contribute."
    )
    htab, atab = st.tabs([f"🏠 {home_team} · HOME only ({len(home_df)})", f"✈️ {away_team} · AWAY only ({len(away_df)})"])
    display_cols = [
        "date", "source_file", "opponent", "result", "gf", "ga", "xg", "xga",
        "shots_for", "shots_against", "sot_for", "sot_against", "corners_for", "corners_against",
        "possession", "cards_for", "cards_against", "fouls_for", "fouls_against", "weight", "parse_quality",
    ]
    with htab:
        st.dataframe(home_df[[c for c in display_cols if c in home_df.columns]].sort_values("date", ascending=False), hide_index=True, use_container_width=True, height=520)
        st.download_button(
            f"Download {home_team} home-only parsed CSV",
            home_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{home_team.replace(' ', '_')}_home_only.csv",
            mime="text/csv",
        )
    with atab:
        st.dataframe(away_df[[c for c in display_cols if c in away_df.columns]].sort_values("date", ascending=False), hide_index=True, use_container_width=True, height=520)
        st.download_button(
            f"Download {away_team} away-only parsed CSV",
            away_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{away_team.replace(' ', '_')}_away_only.csv",
            mime="text/csv",
        )

with analytics_tab:
    st.subheader("Venue-specific performance comparison")
    if result is None:
        st.info("Run a prediction to calculate the engine's attack/defence indices and derived features.")
    else:
        comp = feature_comparison(result)
        st.dataframe(comp.round(3), hide_index=True, use_container_width=True)
        h, a = result["home_features"], result["away_features"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{home_team} attack strength", fmt(h["attack_strength"], 1))
        m2.metric(f"{home_team} defence strength", fmt(h["defense_strength"], 1))
        m3.metric(f"{away_team} attack strength", fmt(a["attack_strength"], 1))
        m4.metric(f"{away_team} defence strength", fmt(a["defense_strength"], 1))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### {home_team} home profile")
            st.dataframe(pd.DataFrame([
                {"Metric": "Attack strength", "Value": h["attack_strength"]},
                {"Metric": "Attack weakness", "Value": h["attack_weakness"]},
                {"Metric": "Defence strength", "Value": h["defense_strength"]},
                {"Metric": "Defence weakness", "Value": h["defense_weakness"]},
                {"Metric": "PPG", "Value": h["ppg"]},
                {"Metric": "xG difference", "Value": h["xgd"]},
            ]).round(3), hide_index=True, use_container_width=True)
        with c2:
            st.markdown(f"#### {away_team} away profile")
            st.dataframe(pd.DataFrame([
                {"Metric": "Attack strength", "Value": a["attack_strength"]},
                {"Metric": "Attack weakness", "Value": a["attack_weakness"]},
                {"Metric": "Defence strength", "Value": a["defense_strength"]},
                {"Metric": "Defence weakness", "Value": a["defense_weakness"]},
                {"Metric": "PPG", "Value": a["ppg"]},
                {"Metric": "xG difference", "Value": a["xgd"]},
            ]).round(3), hide_index=True, use_container_width=True)

with models_tab:
    st.subheader("12-method ensemble")
    if result is None:
        st.info("Run a prediction first.")
    else:
        model_table = result["models"].copy()
        for col in ["Home", "Draw", "Away", "BaseWeight", "QualityFactor", "Weight"]:
            if col in model_table.columns:
                model_table[col] = model_table[col].map(
                    lambda x: round(100 * float(x), 2)
                    if col in {"Home", "Draw", "Away", "Weight"}
                    else round(float(x), 3)
                )
        model_table = model_table.rename(columns={"Home": "Home %", "Draw": "Draw %", "Away": "Away %", "Weight": "Ensemble weight %"})
        st.dataframe(model_table, hide_index=True, use_container_width=True)
        st.caption(
            "The models consume only the two parsed venue-specific tables. xG-based methods are automatically down-weighted if xG coverage is incomplete."
        )

        st.markdown("### Corners model ensemble")
        corners = result["corners"]["breakdown"].copy()
        corners["P_Over_8_5"] = corners["P_Over_8_5"].map(lambda x: round(100 * float(x), 2))
        st.dataframe(corners, hide_index=True, use_container_width=True)
