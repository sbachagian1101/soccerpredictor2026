import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

COMPLETED_STATUSES = {'complete', 'completed', 'finished', 'ft', 'full time'}
UPCOMING_STATUSES = {'incomplete', 'scheduled', 'not started', 'ns', 'upcoming'}

COLUMN_ALIASES = {
    'timestamp': ['timestamp', 'unix_timestamp', 'kickoff_timestamp'],
    'date_GMT': ['date_GMT', 'date', 'Date', 'match_date', 'kickoff'],
    'status': ['status', 'Status'],
    'home_team_name': ['home_team_name', 'HomeTeam', 'home_team', 'Home Team', 'home'],
    'away_team_name': ['away_team_name', 'AwayTeam', 'away_team', 'Away Team', 'away'],
    'home_team_goal_count': ['home_team_goal_count', 'FTHG', 'home_goals', 'HomeGoals', 'HG'],
    'away_team_goal_count': ['away_team_goal_count', 'FTAG', 'away_goals', 'AwayGoals', 'AG'],
    'team_a_xg': ['team_a_xg', 'home_xg', 'Home xG', 'xG_Home'],
    'team_b_xg': ['team_b_xg', 'away_xg', 'Away xG', 'xG_Away'],
    'home_team_shots': ['home_team_shots', 'HS', 'home_shots'],
    'away_team_shots': ['away_team_shots', 'AS', 'away_shots'],
    'home_team_shots_on_target': ['home_team_shots_on_target', 'HST', 'home_sot', 'home_shots_on_target'],
    'away_team_shots_on_target': ['away_team_shots_on_target', 'AST', 'away_sot', 'away_shots_on_target'],
}

REQUIRED_CANONICAL = {
    'timestamp', 'date_GMT', 'status', 'home_team_name', 'away_team_name',
    'home_team_goal_count', 'away_team_goal_count'
}

OPTIONAL_NUMERIC = [
    'team_a_xg', 'team_b_xg', 'home_team_shots', 'away_team_shots',
    'home_team_shots_on_target', 'away_team_shots_on_target'
]

@dataclass
class PredictionResult:
    home_team: str
    away_team: str
    kickoff: str
    lambda_home: float
    lambda_away: float
    p_home: float
    p_draw: float
    p_away: float
    poisson_probs: Tuple[float, float, float]
    elo_probs: Tuple[float, float, float]
    form_probs: Tuple[float, float, float]
    strength_probs: Tuple[float, float, float]
    home_attack: float
    away_attack: float
    home_defense: float
    away_defense: float
    home_def_weakness: float
    away_def_weakness: float
    home_attack_strength: float
    away_attack_strength: float
    home_attack_weakness: float
    away_attack_weakness: float
    home_defense_strength: float
    away_defense_strength: float
    home_defense_weakness: float
    away_defense_weakness: float
    home_attack_components: Dict[str, float]
    away_attack_components: Dict[str, float]
    home_defense_components: Dict[str, float]
    away_defense_components: Dict[str, float]
    top_scores: List[Tuple[str, float]]
    group_size: int
    training_matches: int
    data_quality: float
    model_weights: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _first_existing(columns, aliases):
    for c in aliases:
        if c in columns:
            return c
    return None


def canonicalize_dataframe(df: pd.DataFrame, source_name: str = 'CSV') -> pd.DataFrame:
    """Map common soccer CSV column names to the canonical names used by the model."""
    df = df.copy()
    ren = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _first_existing(df.columns, aliases)
        if found and found != canonical and canonical not in df.columns:
            ren[found] = canonical
    if ren:
        df = df.rename(columns=ren)

    # Build a timestamp from a date column when possible.
    if 'timestamp' not in df.columns and 'date_GMT' in df.columns:
        parsed = pd.to_datetime(df['date_GMT'], errors='coerce', utc=True, dayfirst=False)
        if parsed.notna().any():
            df['timestamp'] = parsed.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)

    # Build a readable date when timestamp exists but date text doesn't.
    if 'date_GMT' not in df.columns and 'timestamp' in df.columns:
        dt = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='s', utc=True, errors='coerce')
        df['date_GMT'] = dt.dt.strftime('%Y-%m-%d %H:%M UTC')

    # Infer status if the source omits it: rows with both goals are treated as completed.
    if 'status' not in df.columns:
        if {'home_team_goal_count', 'away_team_goal_count'}.issubset(df.columns):
            hg = pd.to_numeric(df['home_team_goal_count'], errors='coerce')
            ag = pd.to_numeric(df['away_team_goal_count'], errors='coerce')
            df['status'] = np.where(hg.notna() & ag.notna(), 'complete', 'incomplete')

    missing = REQUIRED_CANONICAL - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} cannot be parsed because these required fields are missing: {sorted(missing)}. "
            "Supported examples include home_team_name/HomeTeam, away_team_name/AwayTeam, "
            "home_team_goal_count/FTHG, away_team_goal_count/FTAG, date/date_GMT and status."
        )

    for c in ['timestamp', 'home_team_goal_count', 'away_team_goal_count'] + OPTIONAL_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['status'] = df['status'].astype(str).str.lower().str.strip()
    df['home_team_name'] = df['home_team_name'].astype(str).str.strip()
    df['away_team_name'] = df['away_team_name'].astype(str).str.strip()
    df = df[(df['home_team_name'] != '') & (df['away_team_name'] != '')]
    df = df[df['timestamp'].notna()].copy()
    return df.sort_values('timestamp').reset_index(drop=True)


def load_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        raw = pd.read_csv(path)
        frames.append(canonicalize_dataframe(raw, source_name=path))
    if not frames:
        raise ValueError('No CSV files loaded.')
    return pd.concat(frames, ignore_index=True, sort=False).sort_values('timestamp').reset_index(drop=True)


def combine_uploaded_frames(named_frames: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    if not named_frames:
        raise ValueError('No CSV files loaded.')
    frames = [canonicalize_dataframe(df, source_name=name) for name, df in named_frames]
    return pd.concat(frames, ignore_index=True, sort=False).sort_values('timestamp').reset_index(drop=True)


def all_teams(df: pd.DataFrame) -> List[str]:
    teams = set(df['home_team_name'].dropna().astype(str)) | set(df['away_team_name'].dropna().astype(str))
    return sorted(t for t in teams if t and t.lower() != 'nan')


def find_upcoming_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df['status'].isin(UPCOMING_STATUSES)].copy()
    cols = ['timestamp', 'date_GMT', 'home_team_name', 'away_team_name', 'status']
    return out.sort_values('timestamp')[cols].reset_index(drop=True)


def resolve_fixture_cutoff(df: pd.DataFrame, home: str, away: str, preferred_timestamp: Optional[float] = None):
    if preferred_timestamp is not None:
        row = df[(df['home_team_name'] == home) & (df['away_team_name'] == away) & (df['timestamp'] == preferred_timestamp)]
        if not row.empty:
            return float(preferred_timestamp), str(row.iloc[0]['date_GMT']), 'selected fixture'
        return float(preferred_timestamp), str(preferred_timestamp), 'manual cutoff'

    upcoming = df[(df['home_team_name'] == home) & (df['away_team_name'] == away) & df['status'].isin(UPCOMING_STATUSES)]
    if not upcoming.empty:
        r = upcoming.sort_values('timestamp').iloc[0]
        return float(r['timestamp']), str(r['date_GMT']), 'upcoming fixture'

    completed = df[df['status'].isin(COMPLETED_STATUSES)]
    if completed.empty:
        raise ValueError('No completed matches are available in the uploaded files.')
    cutoff = float(completed['timestamp'].max()) + 1.0
    label = 'Manual match — latest completed data'
    return cutoff, label, 'latest completed data'


def _connected_group(season_matches: pd.DataFrame, home: str, away: str) -> set:
    adj = defaultdict(set)
    for _, r in season_matches.iterrows():
        a, b = r['home_team_name'], r['away_team_name']
        if not a or not b or a == 'nan' or b == 'nan':
            continue
        adj[a].add(b)
        adj[b].add(a)
    seen = {home}
    stack = [home]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    if away not in seen:
        return set(season_matches['home_team_name']) | set(season_matches['away_team_name'])
    return seen


def _team_rows(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    out = []
    for _, r in matches.sort_values('timestamp').iterrows():
        is_home = r['home_team_name'] == team
        is_away = r['away_team_name'] == team
        if not (is_home or is_away):
            continue
        if is_home:
            row = {
                'timestamp': r['timestamp'], 'date_GMT': r.get('date_GMT', ''), 'venue': 'H',
                'opponent': r['away_team_name'], 'gf': r['home_team_goal_count'], 'ga': r['away_team_goal_count'],
                'xgf': r.get('team_a_xg', np.nan), 'xga': r.get('team_b_xg', np.nan),
                'shots': r.get('home_team_shots', np.nan), 'sot': r.get('home_team_shots_on_target', np.nan),
                'oppshots': r.get('away_team_shots', np.nan), 'oppsot': r.get('away_team_shots_on_target', np.nan),
            }
        else:
            row = {
                'timestamp': r['timestamp'], 'date_GMT': r.get('date_GMT', ''), 'venue': 'A',
                'opponent': r['home_team_name'], 'gf': r['away_team_goal_count'], 'ga': r['home_team_goal_count'],
                'xgf': r.get('team_b_xg', np.nan), 'xga': r.get('team_a_xg', np.nan),
                'shots': r.get('away_team_shots', np.nan), 'sot': r.get('away_team_shots_on_target', np.nan),
                'oppshots': r.get('home_team_shots', np.nan), 'oppsot': r.get('home_team_shots_on_target', np.nan),
            }
        if pd.isna(row['gf']) or pd.isna(row['ga']):
            continue
        row['points'] = 3 if row['gf'] > row['ga'] else 1 if row['gf'] == row['ga'] else 0
        row['result'] = 'W' if row['gf'] > row['ga'] else 'D' if row['gf'] == row['ga'] else 'L'
        for c in ['xgf', 'xga']:
            v = pd.to_numeric(row[c], errors='coerce')
            row[c] = np.nan if pd.isna(v) or v < 0 else float(v)
        for c in ['shots', 'sot', 'oppshots', 'oppsot']:
            v = pd.to_numeric(row[c], errors='coerce')
            row[c] = np.nan if pd.isna(v) or v < 0 else float(v)
        out.append(row)
    return pd.DataFrame(out)


def _league_team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in matches.iterrows():
        axg, bxg = r.get('team_a_xg', np.nan), r.get('team_b_xg', np.nan)
        hs, aw = r.get('home_team_shots', np.nan), r.get('away_team_shots', np.nan)
        hsot, asot = r.get('home_team_shots_on_target', np.nan), r.get('away_team_shots_on_target', np.nan)
        rows.append({'venue': 'H', 'gf': r['home_team_goal_count'], 'ga': r['away_team_goal_count'],
                     'xgf': axg, 'xga': bxg, 'shots': hs, 'sot': hsot, 'oppshots': aw, 'oppsot': asot})
        rows.append({'venue': 'A', 'gf': r['away_team_goal_count'], 'ga': r['home_team_goal_count'],
                     'xgf': bxg, 'xga': axg, 'shots': aw, 'sot': asot, 'oppshots': hs, 'oppsot': hsot})
    league = pd.DataFrame(rows)
    for c in ['gf', 'ga', 'xgf', 'xga', 'shots', 'sot', 'oppshots', 'oppsot']:
        league[c] = pd.to_numeric(league[c], errors='coerce')
    return league


def _safe_mean(s, fallback):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return float(s.mean()) if not s.empty else float(fallback)


def _safe_ratio(x, base, lo=0.35, hi=2.50):
    if pd.isna(x) or pd.isna(base) or base <= 0:
        return 1.0
    return float(np.clip(float(x) / float(base), lo, hi))


def _blend_metric(tm: pd.DataFrame, venue: str, col: str, overall_base: float, venue_base: float) -> float:
    # Explicitly includes season, venue, last-5 overall and last-5 venue form.
    pieces, weights = [], []
    groups = [
        (tm, 0.30, overall_base),
        (tm[tm['venue'] == venue], 0.25, venue_base),
        (tm.tail(5), 0.25, overall_base),
        (tm[tm['venue'] == venue].tail(5), 0.20, venue_base),
    ]
    for sub, w, base in groups:
        if sub.empty or col not in sub:
            continue
        vals = pd.to_numeric(sub[col], errors='coerce').dropna()
        if vals.empty:
            continue
        n = len(vals)
        shrunk = (vals.mean() * n + base * 2.0) / (n + 2.0)
        pieces.append(_safe_ratio(shrunk, base))
        weights.append(w)
    return float(np.average(pieces, weights=weights)) if weights else float('nan')


def _weighted_available(values_with_weights):
    """Weighted mean that redistributes weight across metrics that are actually present."""
    valid = [(float(v), float(w)) for v, w in values_with_weights if pd.notna(v) and np.isfinite(v)]
    if not valid:
        return 1.0
    total_w = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / total_w if total_w > 0 else 1.0


def _strengths(group_matches: pd.DataFrame, home: str, away: str):
    league = _league_team_rows(group_matches)
    overall_fallbacks = {'gf': 1.4, 'ga': 1.4, 'xgf': 1.4, 'xga': 1.4, 'sot': 4.5, 'oppsot': 4.5}
    bases = {}
    for col in ['gf', 'ga', 'xgf', 'xga', 'sot', 'oppsot']:
        ob = _safe_mean(league[col], overall_fallbacks[col])
        bases[(col, 'O')] = ob
        bases[(col, 'H')] = _safe_mean(league.loc[league['venue'] == 'H', col], ob)
        bases[(col, 'A')] = _safe_mean(league.loc[league['venue'] == 'A', col], ob)

    result = {}
    for team, venue in [(home, 'H'), (away, 'A')]:
        tm = _team_rows(group_matches, team)
        if tm.empty:
            raise ValueError(f'No completed historical matches were found for {team} before the selected cutoff.')
        gf = _blend_metric(tm, venue, 'gf', bases[('gf', 'O')], bases[('gf', venue)])
        xgf = _blend_metric(tm, venue, 'xgf', bases[('xgf', 'O')], bases[('xgf', venue)])
        sot = _blend_metric(tm, venue, 'sot', bases[('sot', 'O')], bases[('sot', venue)])
        ga = _blend_metric(tm, venue, 'ga', bases[('ga', 'O')], bases[('ga', venue)])
        xga = _blend_metric(tm, venue, 'xga', bases[('xga', 'O')], bases[('xga', venue)])
        oppsot = _blend_metric(tm, venue, 'oppsot', bases[('oppsot', 'O')], bases[('oppsot', venue)])

        # Re-normalise the component weights if an optional metric (xG or SOT) is absent.
        # This is preferable to silently assigning a missing metric an average value.
        attack_ratio = _weighted_available([(gf, 0.55), (xgf, 0.30), (sot, 0.15)])
        defweak_ratio = _weighted_available([(ga, 0.55), (xga, 0.30), (oppsot, 0.15)])

        attack_index = 100.0 * attack_ratio
        defense_index = 100.0 / defweak_ratio if defweak_ratio > 0 else 100.0
        defweak_index = 100.0 * defweak_ratio

        # Strength and weakness are calculated separately from the underlying components.
        # A team can therefore have both (e.g. strong SOT generation but weak goal output).
        attack_parts = [(gf, 0.55), (xgf, 0.30), (sot, 0.15)]
        defense_parts = [(ga, 0.55), (xga, 0.30), (oppsot, 0.15)]

        def directional(parts, favourable_when_high=True):
            valid = [(float(v), float(w)) for v, w in parts if pd.notna(v) and np.isfinite(v)]
            total_w = sum(w for _, w in valid) or 1.0
            if favourable_when_high:
                strength = sum(w * max(v - 1.0, 0.0) for v, w in valid) / total_w
                weakness = sum(w * max(1.0 - v, 0.0) for v, w in valid) / total_w
            else:
                strength = sum(w * max(1.0 - v, 0.0) for v, w in valid) / total_w
                weakness = sum(w * max(v - 1.0, 0.0) for v, w in valid) / total_w
            return 100.0 * strength, 100.0 * weakness

        attack_strength, attack_weakness = directional(attack_parts, favourable_when_high=True)
        defense_strength, defense_weakness = directional(defense_parts, favourable_when_high=False)

        result[team] = {
            'attack_ratio': attack_ratio,
            'defweak_ratio': defweak_ratio,
            'attack_index': attack_index,
            'defense_index': defense_index,
            'defweak_index': defweak_index,
            'attack_strength': attack_strength,
            'attack_weakness': attack_weakness,
            'defense_strength': defense_strength,
            'defense_weakness': defense_weakness,
            'attack_components': {'Goals': gf, 'xG': xgf, 'Shots on target': sot},
            'defense_components': {'Goals conceded': ga, 'xGA': xga, 'Opponent SOT': oppsot},
            'rows': tm,
        }
    return result, bases


def _poisson_probs(lh: float, la: float, max_goals=10, rho=-0.04):
    hp = np.array([math.exp(-lh) * lh ** i / math.factorial(i) for i in range(max_goals + 1)])
    ap = np.array([math.exp(-la) * la ** j / math.factorial(j) for j in range(max_goals + 1)])
    mat = np.outer(hp, ap)
    mat[0, 0] *= (1 - lh * la * rho)
    mat[0, 1] *= (1 + lh * rho)
    mat[1, 0] *= (1 + la * rho)
    mat[1, 1] *= (1 - rho)
    mat /= mat.sum()
    return np.array([np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()]), mat


def _elo_probs(group_matches: pd.DataFrame, home: str, away: str, k=28, home_adv=55):
    teams = set(group_matches['home_team_name']) | set(group_matches['away_team_name'])
    ratings = {t: 1500.0 for t in teams}
    for _, r in group_matches.sort_values('timestamp').iterrows():
        h, a = r['home_team_name'], r['away_team_name']
        rh, ra = ratings[h], ratings[a]
        eh = 1 / (1 + 10 ** (-((rh + home_adv) - ra) / 400))
        score = 1.0 if r['home_team_goal_count'] > r['away_team_goal_count'] else 0.5 if r['home_team_goal_count'] == r['away_team_goal_count'] else 0.0
        margin = abs(r['home_team_goal_count'] - r['away_team_goal_count'])
        km = k * (1 + 0.12 * min(margin, 4))
        ratings[h] += km * (score - eh)
        ratings[a] -= km * (score - eh)
    diff = (ratings.get(home, 1500) + home_adv) - ratings.get(away, 1500)
    home_share = 1 / (1 + 10 ** (-diff / 400))
    draw_rate = float((group_matches['home_team_goal_count'] == group_matches['away_team_goal_count']).mean())
    closeness = math.exp(-abs(diff) / 300)
    pdraw = float(np.clip(draw_rate * (0.82 + 0.30 * closeness), 0.12, 0.34))
    return np.array([(1 - pdraw) * home_share, pdraw, (1 - pdraw) * (1 - home_share)])


def _form_probs(group_matches: pd.DataFrame, home: str, away: str):
    draw_rate = float((group_matches['home_team_goal_count'] == group_matches['away_team_goal_count']).mean())
    league_ppg = 1.35

    def form_score(team, venue):
        tm = _team_rows(group_matches, team)
        def part(sub):
            if sub.empty:
                return 0.0
            ppg = sub['points'].mean()
            gd = (sub['gf'] - sub['ga']).mean()
            xgd = (sub['xgf'] - sub['xga']).dropna()
            xgd = xgd.mean() if not xgd.empty else 0.0
            return 0.60 * ((ppg - league_ppg) / 1.5) + 0.25 * (gd / 2.0) + 0.15 * (xgd / 1.5)
        return 0.60 * part(tm.tail(5)) + 0.40 * part(tm[tm['venue'] == venue].tail(5))

    sh, sa = form_score(home, 'H'), form_score(away, 'A')
    diff = sh - sa + 0.10
    home_share = 1 / (1 + math.exp(-1.15 * diff))
    pdraw = float(np.clip(draw_rate * (1.15 - 0.20 * min(abs(diff), 2)), 0.13, 0.32))
    return np.array([(1 - pdraw) * home_share, pdraw, (1 - pdraw) * (1 - home_share)])


def _strength_logistic_probs(strengths: Dict, home: str, away: str, draw_rate: float):
    # Independent attack-v-defence rating transformed through a logistic function.
    home_net = math.log(max(strengths[home]['attack_ratio'], 0.2)) - math.log(max(strengths[away]['defweak_ratio'], 0.2))
    away_net = math.log(max(strengths[away]['attack_ratio'], 0.2)) - math.log(max(strengths[home]['defweak_ratio'], 0.2))
    diff = (home_net - away_net) + 0.12
    home_share = 1 / (1 + math.exp(-1.35 * diff))
    closeness = math.exp(-abs(diff))
    pdraw = float(np.clip(draw_rate * (0.85 + 0.30 * closeness), 0.12, 0.33))
    return np.array([(1 - pdraw) * home_share, pdraw, (1 - pdraw) * (1 - home_share)])


def team_summary(matches: pd.DataFrame, team: str, venue: str = None, last_n: int = None) -> Dict[str, float]:
    tm = _team_rows(matches, team)
    if venue:
        tm = tm[tm['venue'] == venue]
    if last_n:
        tm = tm.tail(last_n)
    if tm.empty:
        return {'Matches': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'PPG': 0, 'GD/Match': 0, 'xGF': np.nan, 'xGA': np.nan, 'SOT': np.nan, 'Opp SOT': np.nan}
    return {
        'Matches': len(tm),
        'W': int((tm['result'] == 'W').sum()), 'D': int((tm['result'] == 'D').sum()), 'L': int((tm['result'] == 'L').sum()),
        'GF': round(tm['gf'].mean(), 2), 'GA': round(tm['ga'].mean(), 2), 'PPG': round(tm['points'].mean(), 2),
        'GD/Match': round((tm['gf'] - tm['ga']).mean(), 2),
        'xGF': round(tm['xgf'].dropna().mean(), 2) if tm['xgf'].notna().any() else np.nan,
        'xGA': round(tm['xga'].dropna().mean(), 2) if tm['xga'].notna().any() else np.nan,
        'SOT': round(tm['sot'].dropna().mean(), 2) if tm['sot'].notna().any() else np.nan,
        'Opp SOT': round(tm['oppsot'].dropna().mean(), 2) if tm['oppsot'].notna().any() else np.nan,
    }


def recent_matches_table(matches: pd.DataFrame, team: str, venue: str = None, n: int = 5) -> pd.DataFrame:
    tm = _team_rows(matches, team)
    if venue:
        tm = tm[tm['venue'] == venue]
    cols = ['date_GMT', 'venue', 'opponent', 'gf', 'ga', 'result', 'xgf', 'xga', 'sot', 'oppsot']
    if tm.empty:
        return pd.DataFrame(columns=cols)
    return tm.tail(n)[cols].iloc[::-1].reset_index(drop=True)



def _fmt_metric(v, digits=2):
    return 'N/A' if pd.isna(v) else f'{float(v):.{digits}f}'


def _index_label(v: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if v >= 115: return 'very strong'
        if v >= 105: return 'strong'
        if v >= 97: return 'around average'
        if v >= 88: return 'weak'
        return 'very weak'
    if v >= 115: return 'very vulnerable'
    if v >= 105: return 'vulnerable'
    if v >= 97: return 'around average'
    if v >= 88: return 'resistant'
    return 'very resistant'


def build_match_explanation(result: PredictionResult, training: pd.DataFrame) -> List[str]:
    """Create deterministic, match-specific explanation text from the same inputs used by the model."""
    r = result
    h_over = team_summary(training, r.home_team)
    a_over = team_summary(training, r.away_team)
    h5 = team_summary(training, r.home_team, last_n=5)
    a5 = team_summary(training, r.away_team, last_n=5)
    hh5 = team_summary(training, r.home_team, venue='H', last_n=5)
    aa5 = team_summary(training, r.away_team, venue='A', last_n=5)

    outcomes = [(r.home_team, r.p_home), ('Draw', r.p_draw), (r.away_team, r.p_away)]
    pick, pick_p = max(outcomes, key=lambda x: x[1])
    ordered = sorted(outcomes, key=lambda x: x[1], reverse=True)
    margin = (ordered[0][1] - ordered[1][1]) * 100

    model_rows = [
        ('Poisson/Dixon-Coles', r.poisson_probs),
        ('Elo', r.elo_probs),
        ('Recent form', r.form_probs),
        ('Attack-Defence', r.strength_probs),
    ]
    labels = [r.home_team, 'Draw', r.away_team]
    model_picks = [(name, labels[int(np.argmax(probs))], max(probs)) for name, probs in model_rows]
    votes = defaultdict(int)
    for _, mpick, _ in model_picks:
        votes[mpick] += 1
    agreeing = votes.get(pick, 0)

    lines = []
    lines.append(
        f'Overall verdict: {pick} is the model choice at {pick_p*100:.1f}%. '
        f'The lead over the second-most likely outcome is {margin:.1f} percentage points, '
        f'and {agreeing} of the 4 component models independently select the same outcome.'
    )

    atk_gap = r.home_attack - r.away_attack
    if abs(atk_gap) < 3:
        attack_text = 'The two attacks rate very similarly.'
    elif atk_gap > 0:
        attack_text = f'{r.home_team} has the stronger attacking profile by {atk_gap:.1f} index points.'
    else:
        attack_text = f'{r.away_team} has the stronger attacking profile by {abs(atk_gap):.1f} index points.'
    lines.append(
        f'Attack: {r.home_team} has an Attack Index of {r.home_attack:.1f} ({_index_label(r.home_attack)}), '
        f'while {r.away_team} is {r.away_attack:.1f} ({_index_label(r.away_attack)}). {attack_text} '
        f'Across the attack components, {r.home_team} has {r.home_attack_strength:.1f} strength points and {r.home_attack_weakness:.1f} weakness points; '
        f'{r.away_team} has {r.away_attack_strength:.1f} strength points and {r.away_attack_weakness:.1f} weakness points.'
    )

    def_gap = r.home_defense - r.away_defense
    if abs(def_gap) < 3:
        defense_text = 'Their defensive ratings are close.'
    elif def_gap > 0:
        defense_text = f'{r.home_team} has the stronger defence by {def_gap:.1f} index points.'
    else:
        defense_text = f'{r.away_team} has the stronger defence by {abs(def_gap):.1f} index points.'
    lines.append(
        f'Defence: {r.home_team} has a Defence Index of {r.home_defense:.1f} ({_index_label(r.home_defense)}) '
        f'and {r.away_team} {r.away_defense:.1f} ({_index_label(r.away_defense)}). {defense_text} '
        f'Defensive vulnerability is {r.home_def_weakness:.1f} for {r.home_team} and {r.away_def_weakness:.1f} for {r.away_team} '
        f'on an index where 100 is average and higher is worse.'
    )

    lines.append(
        f'Recent overall form: over the last five, {r.home_team} is {h5["W"]}-{h5["D"]}-{h5["L"]} '
        f'({h5["PPG"]:.2f} PPG, GD/match {h5["GD/Match"]:+.2f}); {r.away_team} is '
        f'{a5["W"]}-{a5["D"]}-{a5["L"]} ({a5["PPG"]:.2f} PPG, GD/match {a5["GD/Match"]:+.2f}).'
    )
    lines.append(
        f'Venue form: {r.home_team}\'s last five home matches produced {hh5["PPG"]:.2f} PPG and {hh5["GD/Match"]:+.2f} GD/match; '
        f'{r.away_team}\'s last five away matches produced {aa5["PPG"]:.2f} PPG and {aa5["GD/Match"]:+.2f} GD/match. '
        'This split is explicitly included in the attack/defence and recent-form calculations.'
    )

    xg_bits = []
    if not pd.isna(h5['xGF']) and not pd.isna(h5['xGA']):
        xg_bits.append(f'{r.home_team} last-5 xG {h5["xGF"]:.2f} for / {h5["xGA"]:.2f} against')
    if not pd.isna(a5['xGF']) and not pd.isna(a5['xGA']):
        xg_bits.append(f'{r.away_team} last-5 xG {a5["xGF"]:.2f} for / {a5["xGA"]:.2f} against')
    if xg_bits:
        lines.append('Underlying chance quality: ' + '; '.join(xg_bits) + '.')

    lines.append(
        f'Expected goals: the score model projects {r.lambda_home:.2f} for {r.home_team} and {r.lambda_away:.2f} for {r.away_team}. '
        f'The most likely exact score is {r.top_scores[0][0]} ({r.top_scores[0][1]*100:.1f}%), but exact-score probabilities are naturally much lower than 1X2 probabilities.'
    )

    model_text = '; '.join(f'{name}: {mpick} ({prob*100:.1f}%)' for name, mpick, prob in model_picks)
    lines.append('Model agreement: ' + model_text + '.')

    caution = []
    if r.data_quality < 0.75:
        caution.append(f'data quality is only {r.data_quality*100:.0f}%')
    if agreeing <= 2:
        caution.append('the component models are split')
    if margin < 10:
        caution.append('the top two 1X2 outcomes are relatively close')
    if min(len(_team_rows(training, r.home_team)), len(_team_rows(training, r.away_team))) < 10:
        caution.append('one or both teams have a small historical sample')
    if caution:
        lines.append('Uncertainty: ' + '; '.join(caution) + '. Treat the percentages as model estimates rather than certainties.')
    else:
        lines.append('Uncertainty: the data sample and model agreement are reasonably supportive, but the probabilities are still model estimates rather than certainties.')

    lines.append(
        'Validation status: the current formula uses fixed expert-chosen component and ensemble weights. '
        'Until those weights are tuned and probability-calibrated with walk-forward historical backtesting, the model should be treated as a strong analytical prototype rather than a fully validated forecasting system.'
    )
    return lines

def data_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    completed = df[df['status'].isin(COMPLETED_STATUSES)]
    return {
        'rows': len(df),
        'completed': len(completed),
        'upcoming': len(find_upcoming_fixtures(df)),
        'teams': len(all_teams(df)),
        'xg_available': bool({'team_a_xg', 'team_b_xg'}.issubset(df.columns) and df[['team_a_xg', 'team_b_xg']].notna().any().any()),
        'shots_available': bool({'home_team_shots_on_target', 'away_team_shots_on_target'}.issubset(df.columns) and df[['home_team_shots_on_target', 'away_team_shots_on_target']].notna().any().any()),
    }


def predict_fixture(df: pd.DataFrame, home: str, away: str, kickoff_timestamp: float = None) -> Tuple[PredictionResult, pd.DataFrame]:
    if not home or not away:
        raise ValueError('Select both a home team and an away team.')
    if home == away:
        raise ValueError('Home team and away team must be different.')

    cutoff, kickoff_label, cutoff_mode = resolve_fixture_cutoff(df, home, away, kickoff_timestamp)
    kickoff_year = pd.to_datetime(cutoff, unit='s', utc=True).year
    years = pd.to_datetime(df['timestamp'], unit='s', utc=True, errors='coerce').dt.year

    # Primary training set: same season and connected competition group before kickoff.
    season_df = df[years == kickoff_year].copy()
    group = _connected_group(season_df, home, away)
    completed = season_df[
        season_df['status'].isin(COMPLETED_STATUSES) &
        (season_df['timestamp'] < cutoff) &
        season_df['home_team_name'].isin(group) &
        season_df['away_team_name'].isin(group)
    ].copy()

    # If the season is too young, add older matches involving teams in the same connected group.
    used_fallback = False
    if len(completed) < 24:
        prior = df[
            df['status'].isin(COMPLETED_STATUSES) &
            (df['timestamp'] < cutoff) &
            (df['timestamp'] < season_df['timestamp'].min() if not season_df.empty else True) &
            df['home_team_name'].isin(group) &
            df['away_team_name'].isin(group)
        ].copy().tail(120)
        if not prior.empty:
            completed = pd.concat([prior, completed], ignore_index=True).sort_values('timestamp')
            used_fallback = True

    if len(completed) < 12:
        raise ValueError('Not enough completed historical matches to produce a stable prediction. Upload more match history.')

    strengths, _ = _strengths(completed, home, away)
    league_home_goals = float(pd.to_numeric(completed['home_team_goal_count'], errors='coerce').mean())
    league_away_goals = float(pd.to_numeric(completed['away_team_goal_count'], errors='coerce').mean())

    lh = float(np.clip(league_home_goals * strengths[home]['attack_ratio'] * strengths[away]['defweak_ratio'], 0.25, 4.75))
    la = float(np.clip(league_away_goals * strengths[away]['attack_ratio'] * strengths[home]['defweak_ratio'], 0.25, 4.75))

    poisson, matrix = _poisson_probs(lh, la)
    elo = _elo_probs(completed, home, away)
    form = _form_probs(completed, home, away)
    draw_rate = float((completed['home_team_goal_count'] == completed['away_team_goal_count']).mean())
    strength_model = _strength_logistic_probs(strengths, home, away, draw_rate)

    # Four-model ensemble. The score model remains dominant, with independent challenges from Elo, form and strength balance.
    weights = {'Poisson/Dixon-Coles': 0.60, 'Elo': 0.20, 'Recent Form': 0.15, 'Attack-Defence': 0.05}
    final = weights['Poisson/Dixon-Coles'] * poisson + weights['Elo'] * elo + weights['Recent Form'] * form + weights['Attack-Defence'] * strength_model
    final = final / final.sum()

    scores = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            scores.append((f'{h}-{a}', float(matrix[h, a])))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:8]

    hm = len(_team_rows(completed, home))
    am = len(_team_rows(completed, away))
    xg_cols = {'team_a_xg', 'team_b_xg'}.issubset(completed.columns)
    xg_share = completed[['team_a_xg', 'team_b_xg']].notna().mean().mean() if xg_cols else 0.0
    data_quality = float(np.clip((min(hm, 15) / 15 * 0.35) + (min(am, 15) / 15 * 0.35) + (min(len(completed), 80) / 80 * 0.20) + (xg_share * 0.10), 0, 1))

    notes = [
        'No bookmaker odds are used in the prediction.',
        f'Data cutoff: {cutoff_mode}. Only completed matches before that cutoff are used.',
        'Attack and defensive ratings combine season context, home/away splits, last 5 overall and last 5 at the relevant venue.',
        'The final 1X2 result is an ensemble of Poisson/Dixon-Coles, Elo, recent-form and attack-v-defence models.',
        'Missing optional xG/SOT components are excluded and the remaining attack/defence weights are re-normalised.',
        'Current component and ensemble weights are fixed heuristics; historical walk-forward backtesting is recommended for calibration.',
    ]
    if used_fallback:
        notes.append('The current season had limited history, so older uploaded matches were added as a fallback training sample.')

    result = PredictionResult(
        home_team=home, away_team=away, kickoff=kickoff_label,
        lambda_home=lh, lambda_away=la,
        p_home=float(final[0]), p_draw=float(final[1]), p_away=float(final[2]),
        poisson_probs=tuple(map(float, poisson)), elo_probs=tuple(map(float, elo)),
        form_probs=tuple(map(float, form)), strength_probs=tuple(map(float, strength_model)),
        home_attack=float(strengths[home]['attack_index']), away_attack=float(strengths[away]['attack_index']),
        home_defense=float(strengths[home]['defense_index']), away_defense=float(strengths[away]['defense_index']),
        home_def_weakness=float(strengths[home]['defweak_index']), away_def_weakness=float(strengths[away]['defweak_index']),
        home_attack_strength=float(strengths[home]['attack_strength']), away_attack_strength=float(strengths[away]['attack_strength']),
        home_attack_weakness=float(strengths[home]['attack_weakness']), away_attack_weakness=float(strengths[away]['attack_weakness']),
        home_defense_strength=float(strengths[home]['defense_strength']), away_defense_strength=float(strengths[away]['defense_strength']),
        home_defense_weakness=float(strengths[home]['defense_weakness']), away_defense_weakness=float(strengths[away]['defense_weakness']),
        home_attack_components=dict(strengths[home]['attack_components']), away_attack_components=dict(strengths[away]['attack_components']),
        home_defense_components=dict(strengths[home]['defense_components']), away_defense_components=dict(strengths[away]['defense_components']),
        top_scores=scores, group_size=len(group), training_matches=len(completed), data_quality=data_quality,
        model_weights=weights, notes=notes
    )
    return result, completed
