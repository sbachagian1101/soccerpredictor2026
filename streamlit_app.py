from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from parser_core import apply_observation_weights, detect_focus_team, parse_match_pages, to_team_perspective
from prediction_engine import build_prediction, feature_comparison, score_matrix_dataframe

st.set_page_config(page_title="Soccer Prediction Lab 2026", page_icon="⚽", layout="wide")

st.markdown(
    """
<style>
:root { --card:#111827; --muted:#94a3b8; --green:#22c55e; --amber:#f59e0b; --blue:#38bdf8; }
.stApp { background: radial-gradient(circle at 15% 0%, #172554 0%, #0b1220 24%, #070b12 60%); }
.block-container { padding-top: 1.2rem; max-width: 1500px; }
.hero { padding: 1.25rem 1.4rem; border:1px solid rgba(148,163,184,.20); border-radius:22px; background:linear-gradient(135deg,rgba(30,64,175,.35),rgba(15,23,42,.88)); margin-bottom:1rem; }
.hero h1 { margin:0; font-size:2.25rem; }
.hero p { margin:.35rem 0 0; color:#cbd5e1; }
.pred-card { min-height:142px; padding:1rem; border-radius:20px; border:1px solid rgba(148,163,184,.18); background:linear-gradient(160deg,rgba(30,41,59,.92),rgba(15,23,42,.82)); text-align:center; }
.pred-card.best { border:1px solid rgba(34,197,94,.65); box-shadow:0 0 28px rgba(34,197,94,.10); }
.pred-label { color:#94a3b8; font-size:.85rem; text-transform:uppercase; letter-spacing:.08em; }
.pred-value { font-size:2.25rem; font-weight:800; margin:.25rem 0; }
.pred-name { color:#e2e8f0; font-weight:600; }
.market-card { padding:1rem 1.1rem; border-radius:18px; border:1px solid rgba(148,163,184,.16); background:rgba(15,23,42,.72); }
.badge { display:inline-block; padding:.24rem .55rem; border-radius:999px; background:rgba(56,189,248,.12); color:#7dd3fc; font-size:.78rem; }
.small-note { color:#94a3b8; font-size:.88rem; }
div[data-testid="stMetric"] { background:rgba(15,23,42,.66); border:1px solid rgba(148,163,184,.16); padding:.8rem; border-radius:16px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <span class="badge">FORM-ONLY · NO MARKET ODDS</span>
  <h1>⚽ Soccer Prediction Lab 2026</h1>
  <p>Paste up to 10 recent FootyStats match pages for each team. The app parses the actual post-match data, weights match quality and recency, then runs a multi-model ensemble.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Match setup")
    upcoming_date = st.date_input("Upcoming match date", value=date.today())
    half_life = st.slider("Recency half-life (days)", 20, 120, 45, 5)
    st.caption("A 45-day half-life means a match 45 days older receives half the recency weight of a match played today.")
    st.divider()
    st.markdown("**Weight formula**")
    st.caption("Recency × venue relevance × data completeness × competition importance × opponent-quality adjustment.")
    st.markdown("**Leakage protection**")
    st.caption("Odds Market, Prediction Stats, Who Will Win and duplicated Current Form sections are not used.")

input_tab, parsed_tab, analytics_tab, prediction_tab, models_tab = st.tabs([
    "1 · Match Input", "2 · Parsed Data", "3 · Team Analytics", "4 · Prediction", "5 · Model Breakdown"
])

with input_tab:
    st.subheader("Paste up to 10 matches for each team")
    st.caption("Paste the complete copied FootyStats text for each historical match page into the relevant box. You may paste between 1 and 10 pages together; if more than 10 valid matches are found, the app keeps the 10 most recent for that team.")
    c1, c2 = st.columns(2)
    with c1:
        home_text = st.text_area(
            "Home team — up to 10 matches",
            value=st.session_state.get("raw_home", ""),
            height=500,
            placeholder="Paste up to 10 FootyStats match pages for the home team here…",
            key="home_text_input",
        )
    with c2:
        away_text = st.text_area(
            "Away team — up to 10 matches",
            value=st.session_state.get("raw_away", ""),
            height=500,
            placeholder="Paste up to 10 FootyStats match pages for the away team here…",
            key="away_text_input",
        )

    if st.button("🔎 Parse Match Data", type="primary", use_container_width=True):
        if not home_text.strip() or not away_text.strip():
            st.error("Paste both teams' historical match text before parsing.")
        else:
            raw_h = parse_match_pages(home_text)
            raw_a = parse_match_pages(away_text)
            team_h = detect_focus_team(raw_h)
            team_a = detect_focus_team(raw_a)
            hdf = to_team_perspective(raw_h, team_h, max_matches=10)
            adf = to_team_perspective(raw_a, team_a, max_matches=10)
            if hdf.empty or adf.empty:
                st.error("I could not identify completed 'Final Results' blocks in one or both text boxes.")
            else:
                st.session_state["raw_home"] = home_text
                st.session_state["raw_away"] = away_text
                st.session_state["home_team"] = team_h
                st.session_state["away_team"] = team_a
                st.session_state["home_df"] = hdf
                st.session_state["away_df"] = adf
                st.session_state.pop("prediction", None)
                st.success(f"Parsed {len(hdf)} matches for {team_h} and {len(adf)} matches for {team_a}. Open Parsed Data to review them.")

    st.markdown("#### What is extracted")
    st.write("Score, xG/xGA, shots, shots on target, possession, corners, cards, fouls, offsides, venue, competition and match date. The team-perspective table also calculates W/D/L and later receives observation weights. Missing post-match fields are retained as missing values and automatically reduce that match's data-completeness weight.")


def editor_table(df: pd.DataFrame, key: str):
    show_cols = [
        "include", "date", "venue", "opponent", "competition", "match_type", "importance", "opponent_quality",
        "result", "gf", "ga", "xg", "xga", "shots_for", "shots_against", "sot_for", "sot_against",
        "possession", "corners_for", "corners_against", "parse_quality",
    ]
    work = df[show_cols].copy()
    return st.data_editor(
        work,
        use_container_width=True,
        hide_index=True,
        key=key,
        disabled=[
            "date", "venue", "opponent", "competition", "result", "gf", "ga", "xg", "xga", "shots_for",
            "shots_against", "sot_for", "sot_against", "possession", "corners_for", "corners_against", "parse_quality",
        ],
        column_config={
            "include": st.column_config.CheckboxColumn("Use"),
            "importance": st.column_config.NumberColumn("Importance", min_value=0.2, max_value=1.4, step=0.05, format="%.2f"),
            "opponent_quality": st.column_config.NumberColumn("Opponent quality", min_value=0.7, max_value=1.3, step=0.05, format="%.2f"),
            "parse_quality": st.column_config.ProgressColumn("Parse quality", min_value=0.0, max_value=1.0, format="%.0%%"),
            "date": st.column_config.DateColumn("Date"),
        },
    )


with parsed_tab:
    if "home_df" not in st.session_state or "away_df" not in st.session_state:
        st.info("Parse both teams on the Match Input tab first.")
    else:
        home_team = st.session_state["home_team"]
        away_team = st.session_state["away_team"]
        st.subheader(f"{home_team} vs {away_team} — parser review")
        st.caption("Up to 10 recent matches per team are retained. Only Use, match type/importance and opponent quality are editable. Correct or exclude a questionable match before predicting.")

        l, r = st.columns(2)
        with l:
            st.markdown(f"### 🏠 {home_team}")
            edited_h = editor_table(st.session_state["home_df"], "editor_home")
        with r:
            st.markdown(f"### ✈️ {away_team}")
            edited_a = editor_table(st.session_state["away_df"], "editor_away")

        for edited, base_key in [(edited_h, "home_df"), (edited_a, "away_df")]:
            base = st.session_state[base_key].copy().reset_index(drop=True)
            edited = edited.reset_index(drop=True)
            for c in edited.columns:
                base[c] = edited[c]
            st.session_state[base_key] = base

        wh = apply_observation_weights(st.session_state["home_df"], upcoming_date, "Home", half_life)
        wa = apply_observation_weights(st.session_state["away_df"], upcoming_date, "Away", half_life)

        st.markdown("#### Final observation weights")
        wc1, wc2 = st.columns(2)
        cols = ["date", "opponent", "match_type", "days_ago", "recency_weight", "venue_weight", "importance", "opponent_quality", "weight"]
        with wc1:
            st.dataframe(wh[cols].style.format({"recency_weight":"{:.3f}","venue_weight":"{:.2f}","importance":"{:.2f}","opponent_quality":"{:.2f}","weight":"{:.3f}"}), use_container_width=True, hide_index=True)
        with wc2:
            st.dataframe(wa[cols].style.format({"recency_weight":"{:.3f}","venue_weight":"{:.2f}","importance":"{:.2f}","opponent_quality":"{:.2f}","weight":"{:.3f}"}), use_container_width=True, hide_index=True)

        if st.button("⚽ PREDICT MATCH", type="primary", use_container_width=True):
            try:
                st.session_state["weighted_home"] = wh
                st.session_state["weighted_away"] = wa
                st.session_state["prediction"] = build_prediction(wh, wa)
                st.success("Prediction complete. Open the Prediction tab.")
            except Exception as exc:
                st.error(f"Prediction error: {exc}")


with analytics_tab:
    if "prediction" not in st.session_state:
        st.info("Parse the data and press PREDICT MATCH first.")
    else:
        res = st.session_state["prediction"]
        h = res["home_features"]; a = res["away_features"]
        st.subheader("Weighted team analytics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric(f"{res['home_team']} Attack", f"{h['attack_strength']:.0f}/100")
        c2.metric(f"{res['home_team']} Defence", f"{h['defense_strength']:.0f}/100")
        c3.metric(f"{res['away_team']} Attack", f"{a['attack_strength']:.0f}/100")
        c4.metric(f"{res['away_team']} Defence", f"{a['defense_strength']:.0f}/100")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Home attack weakness", f"{h['attack_weakness']:.0f}/100")
        c2.metric("Home defence weakness", f"{h['defense_weakness']:.0f}/100")
        c3.metric("Away attack weakness", f"{a['attack_weakness']:.0f}/100")
        c4.metric("Away defence weakness", f"{a['defense_weakness']:.0f}/100")

        left, right = st.columns([1.15, 1])
        with left:
            comp = feature_comparison(res)
            st.dataframe(comp.style.format({res['home_team']:"{:.2f}", res['away_team']:"{:.2f}"}), use_container_width=True, hide_index=True)
        with right:
            cats = ["Attack", "Defence", "Chance creation", "Finishing", "Corner pressure", "Form"]
            hv = [h['attack_strength'], h['defense_strength'], np.clip(50+15*h['xgd'],1,99), np.clip(50+30*h['finishing_delta'],1,99), np.clip(50+7*(h['corners_for']-h['corners_against']),1,99), np.clip(33*h['ppg'],1,99)]
            av = [a['attack_strength'], a['defense_strength'], np.clip(50+15*a['xgd'],1,99), np.clip(50+30*a['finishing_delta'],1,99), np.clip(50+7*(a['corners_for']-a['corners_against']),1,99), np.clip(33*a['ppg'],1,99)]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=hv+[hv[0]], theta=cats+[cats[0]], fill='toself', name=res['home_team']))
            fig.add_trace(go.Scatterpolar(r=av+[av[0]], theta=cats+[cats[0]], fill='toself', name=res['away_team']))
            fig.update_layout(height=430, polar=dict(radialaxis=dict(visible=True, range=[0,100])), margin=dict(l=25,r=25,t=35,b=25), legend=dict(orientation='h'))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Parsed-match xG trend (up to 10 matches)")
        fig = go.Figure()
        for label, df in [(res['home_team'], st.session_state['weighted_home']), (res['away_team'], st.session_state['weighted_away'])]:
            d = df.sort_values('date')
            fig.add_trace(go.Scatter(x=d.date, y=d.xg, mode='lines+markers', name=f'{label} xG'))
            fig.add_trace(go.Scatter(x=d.date, y=d.xga, mode='lines+markers', name=f'{label} xGA', line=dict(dash='dot')))
        fig.update_layout(height=360, yaxis_title='Expected goals', margin=dict(l=20,r=20,t=25,b=20), legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)


with prediction_tab:
    if "prediction" not in st.session_state:
        st.info("Your prediction dashboard will appear here after you press PREDICT MATCH on Parsed Data.")
    else:
        r = st.session_state["prediction"]
        probs = {r['home_team']:r['p_home'], 'Draw':r['p_draw'], r['away_team']:r['p_away']}
        best = max(probs, key=probs.get)
        st.subheader(f"{r['home_team']}  vs  {r['away_team']}")
        st.caption(f"Ensemble expected goals: {r['expected_home_goals']:.2f} – {r['expected_away_goals']:.2f} · Confidence {r['confidence']:.0f}/100")

        cols = st.columns(3)
        cards = [(r['home_team'], '1 · HOME', r['p_home']), ('Draw', 'X · DRAW', r['p_draw']), (r['away_team'], '2 · AWAY', r['p_away'])]
        for col,(name,label,p) in zip(cols,cards):
            cls = 'pred-card best' if name == best else 'pred-card'
            col.markdown(f"<div class='{cls}'><div class='pred-label'>{label}</div><div class='pred-value'>{p*100:.1f}%</div><div class='pred-name'>{name}</div></div>", unsafe_allow_html=True)

        st.success(f"**Primary 1X2 pick: {r['pick']}** · model confidence {r['confidence']:.0f}/100")

        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(f"<div class='market-card'><div class='pred-label'>BTTS</div><div class='pred-value'>{r['btts_yes']*100:.1f}%</div><div class='pred-name'>YES</div><div class='small-note'>No {r['btts_no']*100:.1f}%</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='market-card'><div class='pred-label'>TOTAL GOALS 2.5</div><div class='pred-value'>{r['over25']*100:.1f}%</div><div class='pred-name'>OVER 2.5</div><div class='small-note'>Under {r['under25']*100:.1f}%</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='market-card'><div class='pred-label'>CORNERS 8.5</div><div class='pred-value'>{r['corners']['over85']*100:.1f}%</div><div class='pred-name'>OVER 8.5</div><div class='small-note'>Expected {r['corners']['expected']:.1f}</div></div>", unsafe_allow_html=True)
        top_score = r['top_scores'][0]
        m4.markdown(f"<div class='market-card'><div class='pred-label'>TOP SCORE</div><div class='pred-value'>{top_score[0]}</div><div class='pred-name'>{top_score[1]*100:.1f}%</div><div class='small-note'>Distribution ensemble</div></div>", unsafe_allow_html=True)

        st.markdown("### Full-time score probability matrix (%)")
        sm = score_matrix_dataframe(r, 6)
        heat = go.Figure(data=go.Heatmap(z=sm.values, x=sm.columns, y=sm.index, text=np.round(sm.values,1), texttemplate='%{text:.1f}', hovertemplate='Home %{y} - Away %{x}: %{z:.2f}%<extra></extra>'))
        heat.update_layout(height=530, xaxis_title=f"{r['away_team']} goals", yaxis_title=f"{r['home_team']} goals", margin=dict(l=40,r=20,t=20,b=45))
        st.plotly_chart(heat, use_container_width=True)

        st.markdown("#### Most likely scorelines")
        ts = pd.DataFrame(r['top_scores'], columns=['Score','Probability'])
        ts['Probability'] = (ts['Probability']*100).map(lambda v:f'{v:.2f}%')
        st.dataframe(ts, use_container_width=True, hide_index=True)


with models_tab:
    if "prediction" not in st.session_state:
        st.info("Run a prediction to see the mathematical model breakdown.")
    else:
        r = st.session_state["prediction"]
        st.subheader("12-model 1X2 ensemble")
        md = r['models'].copy()
        md['Home %'] = md['Home']*100; md['Draw %'] = md['Draw']*100; md['Away %'] = md['Away']*100; md['Weight %'] = md['Weight']*100
        st.dataframe(md[['Model','Family','Home %','Draw %','Away %','Weight %']].style.format({'Home %':'{:.1f}','Draw %':'{:.1f}','Away %':'{:.1f}','Weight %':'{:.1f}'}), use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Home', x=md.Model, y=md['Home %']))
        fig.add_trace(go.Bar(name='Draw', x=md.Model, y=md['Draw %']))
        fig.add_trace(go.Bar(name='Away', x=md.Model, y=md['Away %']))
        fig.update_layout(barmode='group', height=470, yaxis_title='Probability %', xaxis_tickangle=-35, margin=dict(l=20,r=20,t=25,b=145), legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Corner model breakdown")
        cb = r['corners']['breakdown'].copy()
        cb['Over 8.5 %'] = cb.P_Over_8_5*100; cb['Weight %'] = cb.Weight*100
        st.dataframe(cb[['Model','Over 8.5 %','Expected_Corners','Weight %']].style.format({'Over 8.5 %':'{:.1f}','Expected_Corners':'{:.2f}','Weight %':'{:.1f}'}), use_container_width=True, hide_index=True)

        st.markdown("### Methods used")
        st.markdown("""
1. Independent Poisson using recent goals.
2. xG Poisson.
3. Attack–defence strength Poisson.
4. Dixon–Coles low-score correction.
5. Bivariate Poisson shared-score component.
6. Negative Binomial goal model.
7. Bayesian Gamma–Poisson posterior predictive model.
8. Skellam goal-difference model.
9. Elo-style recent-performance rating.
10. Composite attack/defence Power Index.
11. Bayesian W/D/L form posterior.
12. Recency-weighted bootstrap simulation.

BTTS, O/U2.5 and the correct-score matrix are calculated from the ensemble of full score-distribution models. Corners O/U8.5 uses a separate five-model ensemble (Poisson, Negative Binomial, Gamma–Poisson, bootstrap and a pressure-adjusted model).
""")
        st.caption("This is a form/performance forecasting system, not a guarantee of match outcome. No bookmaker prices are used by the prediction engine.")
