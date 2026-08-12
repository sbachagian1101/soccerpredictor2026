import io
import pandas as pd
import streamlit as st

from model_core import (
    all_teams, build_match_explanation, combine_uploaded_frames, data_diagnostics, find_upcoming_fixtures,
    predict_fixture, recent_matches_table, team_summary
)

st.set_page_config(page_title='Soccer Predictor', page_icon='⚽', layout='wide')
st.title('⚽ Soccer Predictor')
st.caption('Upload match-history CSV files, choose Home vs Away, and calculate multi-model 1X2 probabilities.')

uploaded = st.sidebar.file_uploader('Upload CSV file(s)', type=['csv'], accept_multiple_files=True)
st.sidebar.caption('You can upload multiple seasons. The model only uses completed matches available before the selected fixture/cutoff.')

@st.cache_data(show_spinner=False)
def parse_files(payload):
    named=[]
    for name, data in payload:
        named.append((name, pd.read_csv(io.BytesIO(data))))
    return combine_uploaded_frames(named)

if not uploaded:
    st.info('Upload one or more soccer match-history CSV files to begin.')
    st.stop()

try:
    df = parse_files([(f.name, f.getvalue()) for f in uploaded])
except Exception as e:
    st.error(f'CSV parsing error: {e}')
    st.stop()

d = data_diagnostics(df)
m1,m2,m3,m4 = st.columns(4)
m1.metric('Rows parsed', d['rows'])
m2.metric('Completed matches', d['completed'])
m3.metric('Teams detected', d['teams'])
m4.metric('Upcoming fixtures', d['upcoming'])
st.caption(f"Optional data detected — xG: {'Yes' if d['xg_available'] else 'No'} · shots on target: {'Yes' if d['shots_available'] else 'No'}")

teams = all_teams(df)
fixtures = find_upcoming_fixtures(df)
fixture_options = ['Manual team selection']
fixture_lookup = {}
for _, r in fixtures.iterrows():
    label = f"{r['date_GMT']} | {r['home_team_name']} vs {r['away_team_name']}"
    fixture_options.append(label)
    fixture_lookup[label] = (r['home_team_name'], r['away_team_name'], float(r['timestamp']))

st.subheader('Select match')
fixture_choice = st.selectbox('Upcoming fixture (optional — auto-fills teams)', fixture_options)

if fixture_choice in fixture_lookup:
    default_home, default_away, fixture_ts = fixture_lookup[fixture_choice]
else:
    default_home = teams[0] if teams else ''
    default_away = teams[1] if len(teams) > 1 else default_home
    fixture_ts = None

left, right = st.columns(2)
with left:
    home_idx = teams.index(default_home) if default_home in teams else 0
    home = st.selectbox('Home team', teams, index=home_idx, key=f'home_{fixture_choice}')
with right:
    away_idx = teams.index(default_away) if default_away in teams else min(1, len(teams)-1)
    away = st.selectbox('Away team', teams, index=away_idx, key=f'away_{fixture_choice}')

# Only preserve fixture timestamp if the user kept the auto-filled pairing.
selected_ts = fixture_ts if fixture_choice in fixture_lookup and (home, away) == fixture_lookup[fixture_choice][:2] else None

if st.button('Analyse match', type='primary', use_container_width=True):
    st.session_state['run_prediction'] = (home, away, selected_ts)

if 'run_prediction' not in st.session_state:
    st.stop()

home, away, selected_ts = st.session_state['run_prediction']
try:
    r, training = predict_fixture(df, home, away, selected_ts)
except Exception as e:
    st.error(str(e))
    st.stop()

pick = max([(r.home_team,r.p_home),('Draw',r.p_draw),(r.away_team,r.p_away)], key=lambda x:x[1])
quality = 'High' if r.data_quality >= .75 else 'Moderate' if r.data_quality >= .50 else 'Limited'

st.divider()
st.subheader(f'{r.home_team} vs {r.away_team}')
st.caption(f'{r.kickoff} · {r.training_matches} historical matches · {r.group_size}-team comparison group · data quality {quality} ({r.data_quality*100:.0f}%)')

c1,c2,c3 = st.columns(3)
c1.metric(f'1 — {r.home_team}', f'{r.p_home*100:.1f}%')
c2.metric('X — Draw', f'{r.p_draw*100:.1f}%')
c3.metric(f'2 — {r.away_team}', f'{r.p_away*100:.1f}%')
st.success(f'Prediction: **{pick[0]}** ({pick[1]*100:.1f}%) · Expected goals: **{r.lambda_home:.2f} – {r.lambda_away:.2f}**')

explain_tab, pred_tab, strength_tab, form_tab, score_tab, method_tab = st.tabs([
    'Explanation','Model breakdown','Strength / Weakness','Recent form','Scorelines','Method'
])

with explain_tab:
    st.markdown('### Why the model produced this prediction')
    for i, line in enumerate(build_match_explanation(r, training), start=1):
        if i == 1:
            st.info(line)
        elif line.startswith('Uncertainty:') or line.startswith('Validation status:'):
            st.warning(line)
        else:
            st.write('• ' + line)


with pred_tab:
    comp = pd.DataFrame({
        'Model':['Poisson / Dixon-Coles','Elo','Recent form','Attack-Defence','Final ensemble'],
        f'1 — {r.home_team}':[r.poisson_probs[0],r.elo_probs[0],r.form_probs[0],r.strength_probs[0],r.p_home],
        'X — Draw':[r.poisson_probs[1],r.elo_probs[1],r.form_probs[1],r.strength_probs[1],r.p_draw],
        f'2 — {r.away_team}':[r.poisson_probs[2],r.elo_probs[2],r.form_probs[2],r.strength_probs[2],r.p_away],
    })
    for c in comp.columns[1:]:
        comp[c] = (comp[c]*100).round(2).astype(str) + '%'
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.markdown('**Ensemble:** ' + ' · '.join(f'{k} {v*100:.0f}%' for k,v in r.model_weights.items()))

with strength_tab:
    strength = pd.DataFrame([
        {
            'Team':r.home_team,'Role':'Home',
            'Attack Index':round(r.home_attack,1),'Defence Index':round(r.home_defense,1),
            'Attacking Strength':round(r.home_attack_strength,1),'Attacking Weakness':round(r.home_attack_weakness,1),
            'Defensive Strength':round(r.home_defense_strength,1),'Defensive Weakness':round(r.home_defense_weakness,1),
            'Defensive Weakness Index':round(r.home_def_weakness,1)
        },
        {
            'Team':r.away_team,'Role':'Away',
            'Attack Index':round(r.away_attack,1),'Defence Index':round(r.away_defense,1),
            'Attacking Strength':round(r.away_attack_strength,1),'Attacking Weakness':round(r.away_attack_weakness,1),
            'Defensive Strength':round(r.away_defense_strength,1),'Defensive Weakness':round(r.away_defense_weakness,1),
            'Defensive Weakness Index':round(r.away_def_weakness,1)
        },
    ])
    st.dataframe(strength, use_container_width=True, hide_index=True)
    st.caption('Indices are centred on 100. Attack/Defence Index >100 is better. Strength and weakness are separate weighted component-deviation points versus the comparison-group average, so a team can show both strengths and weaknesses at the same time.')

    st.markdown('#### Component ratings')
    components = pd.DataFrame([
        {'Team':r.home_team,'Side':'Attack',**{k:(round(v*100,1) if pd.notna(v) else None) for k,v in r.home_attack_components.items()}},
        {'Team':r.away_team,'Side':'Attack',**{k:(round(v*100,1) if pd.notna(v) else None) for k,v in r.away_attack_components.items()}},
        {'Team':r.home_team,'Side':'Defensive vulnerability',**{k:(round(v*100,1) if pd.notna(v) else None) for k,v in r.home_defense_components.items()}},
        {'Team':r.away_team,'Side':'Defensive vulnerability',**{k:(round(v*100,1) if pd.notna(v) else None) for k,v in r.away_defense_components.items()}},
    ])
    st.dataframe(components, use_container_width=True, hide_index=True)
    st.caption('Component values are league-relative ratios ×100. For attack components, higher is stronger. For defensive-vulnerability components, higher is worse.')

    summary=[]
    for team, venue in [(r.home_team,'H'),(r.away_team,'A')]:
        for label,v,n in [('Overall',None,None),('Last 5 overall',None,5),(f'Last 5 {"home" if venue=="H" else "away"}',venue,5)]:
            summary.append({'Team':team,'Split':label,**team_summary(training,team,venue=v,last_n=n)})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

with form_tab:
    l,rcol = st.columns(2)
    with l:
        st.markdown(f'#### {r.home_team} — last 5 overall')
        st.dataframe(recent_matches_table(training,r.home_team,n=5),use_container_width=True,hide_index=True)
        st.markdown(f'#### {r.home_team} — last 5 home')
        st.dataframe(recent_matches_table(training,r.home_team,venue='H',n=5),use_container_width=True,hide_index=True)
    with rcol:
        st.markdown(f'#### {r.away_team} — last 5 overall')
        st.dataframe(recent_matches_table(training,r.away_team,n=5),use_container_width=True,hide_index=True)
        st.markdown(f'#### {r.away_team} — last 5 away')
        st.dataframe(recent_matches_table(training,r.away_team,venue='A',n=5),use_container_width=True,hide_index=True)

with score_tab:
    score_df = pd.DataFrame(r.top_scores,columns=['Score','Probability'])
    score_df['Probability'] = (score_df['Probability']*100).round(2).astype(str)+'%'
    st.dataframe(score_df,use_container_width=True,hide_index=True)

with method_tab:
    st.markdown('''
### Data extraction
The app maps common soccer CSV fields into a standard schema. Required information is match date/timestamp, home team, away team and full-time goals. **xG and shots on target are optional** and improve the attack/defence estimate when present.

### Attack and defence ratings
**Attack** = 55% goals scored + 30% xG + 15% shots on target.  
**Defensive vulnerability** = 55% goals conceded + 30% xGA + 15% opponent shots on target.

If an optional xG/SOT field is unavailable, its weight is **redistributed across the metrics that are present** rather than pretending the missing value is league-average.

Each statistic combines:
- 30% season/available-history overall
- 25% home/away split
- 25% last 5 overall
- 20% last 5 at the relevant venue

Small samples are shrunk toward competition average.

### Prediction models
1. **Poisson / Dixon-Coles** — estimates expected goals and scoreline probabilities.
2. **Elo** — rates opponent-adjusted team strength and home advantage.
3. **Recent form** — uses last-5 PPG, goal difference and xG difference.
4. **Attack-Defence logistic model** — independently compares attacking strength against defensive vulnerability.

The final 1X2 probability blends them at **60% / 20% / 15% / 5%** respectively.

### Strength / weakness interpretation
- **Attack Index**: 100 = group average; above 100 means a stronger attack.
- **Defence Index**: 100 = group average; above 100 means a stronger defence.
- **Attacking Strength / Weakness**: weighted favourable/unfavourable deviations across the attack components relative to average.
- **Defensive Strength / Weakness**: weighted favourable/unfavourable deviations across the defensive components relative to average.

### Validation status
This version is a **multi-model analytical model**, but its 55/30/15 feature weights, recency weights and 50/20/20/10 ensemble weights are still fixed expert-chosen values. They have not yet been optimised by walk-forward backtesting. For a genuinely calibrated forecasting model, the next step is to backtest historical fixtures and tune the weights against **log loss, Brier score and 1X2 calibration**, using only information available before each historical kickoff.

### Data leakage protection
If the selected match is an upcoming fixture contained in the uploaded CSV, only completed matches before its kickoff are used. For a manual pairing, the model uses the latest completed data available.

**Bookmaker odds are not used.**
''')
    st.markdown('**Model notes**')
    for note in r.notes:
        st.write('• ' + note)
