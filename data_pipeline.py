from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/matches/resources/fixtures.csv"

LEAGUES: Dict[str, str] = {
    "E0": "England - Premier League",
    "E1": "England - Championship",
    "E2": "England - League One",
    "E3": "England - League Two",
    "D1": "Germany - Bundesliga",
    "D2": "Germany - 2. Bundesliga",
    "I1": "Italy - Serie A",
    "I2": "Italy - Serie B",
    "SP1": "Spain - La Liga",
    "SP2": "Spain - Segunda Division",
    "F1": "France - Ligue 1",
    "F2": "France - Ligue 2",
    "N1": "Netherlands - Eredivisie",
    "B1": "Belgium - First Division A",
    "P1": "Portugal - Primeira Liga",
    "T1": "Turkey - Super Lig",
    "G1": "Greece - Super League",
}

RAW_STATS = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]

FEATURE_COLUMNS = [
    "home_matches", "away_matches",
    "home_ppg_l5", "away_ppg_l5", "home_ppg_l10", "away_ppg_l10",
    "home_adj_ppg_l5", "away_adj_ppg_l5", "home_wppg_l5", "away_wppg_l5",
    "home_gf_l5", "away_gf_l5", "home_ga_l5", "away_ga_l5",
    "home_gf_l10", "away_gf_l10", "home_ga_l10", "away_ga_l10",
    "home_gd_l10", "away_gd_l10", "home_btts_l10", "away_btts_l10",
    "home_o25_l10", "away_o25_l10", "home_shots_l5", "away_shots_l5",
    "home_sot_l5", "away_sot_l5", "home_corners_l5", "away_corners_l5",
    "home_opp_shots_l5", "away_opp_shots_l5", "home_opp_sot_l5", "away_opp_sot_l5",
    "home_opp_corners_l5", "away_opp_corners_l5", "home_venue_ppg_l5", "away_venue_ppg_l5",
    "home_venue_ppg_l10", "away_venue_ppg_l10", "home_venue_gf_l10", "away_venue_gf_l10",
    "home_venue_ga_l10", "away_venue_ga_l10", "home_venue_sot_l5", "away_venue_sot_l5",
    "home_venue_corners_l5", "away_venue_corners_l5", "elo_home", "elo_away", "elo_diff",
    "elo_home_expectancy", "home_rest_days", "away_rest_days", "league_home_goals",
    "league_away_goals", "league_draw_rate", "league_btts_rate", "league_o25_rate",
    "league_corners_avg", "home_attack_strength", "away_attack_strength", "home_def_weakness",
    "away_def_weakness", "lambda_form_home", "lambda_form_away", "ppg_diff_l5", "ppg_diff_l10",
    "gf_diff_l10", "ga_diff_l10", "gd_diff_l10", "sot_diff_l5", "corners_diff_l5",
    "venue_ppg_diff_l10", "attack_strength_diff", "def_weakness_diff",
]


def current_season_start(today: Optional[datetime] = None) -> int:
    today = today or datetime.now()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def season_label(start_year: int) -> str:
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def default_seasons(n: int = 5) -> List[int]:
    current = current_season_start()
    return list(range(current - n + 1, current + 1))


def football_data_url(league_code: str, start_year: int) -> str:
    return BASE_URL.format(season=season_code(start_year), league=league_code)


def _download_bytes(url: str, timeout: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 SoccerPredictionLab/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def _parse_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return parsed


def canonicalize_football_data(df: pd.DataFrame, league_code: Optional[str] = None,
                               season_start: Optional[int] = None) -> pd.DataFrame:
    out = df.copy()
    rename = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns and v not in out.columns})
    missing = {"Date", "HomeTeam", "AwayTeam"} - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "Div" not in out.columns:
        out["Div"] = league_code or "CUSTOM"
    out["league_code"] = league_code if league_code else out["Div"].astype(str)
    out["season_start"] = season_start if season_start is not None else np.nan
    out["Date"] = _parse_date_series(out["Date"])
    if "Time" not in out.columns:
        out["Time"] = ""
    for col in ["FTHG", "FTAG"] + RAW_STATS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "FTR" not in out.columns:
        out["FTR"] = np.where(
            out["FTHG"].notna() & out["FTAG"].notna(),
            np.where(out["FTHG"] > out["FTAG"], "H", np.where(out["FTHG"] < out["FTAG"], "A", "D")),
            np.nan,
        )
    out["HomeTeam"] = out["HomeTeam"].astype(str).str.strip()
    out["AwayTeam"] = out["AwayTeam"].astype(str).str.strip()
    out = out[out["Date"].notna() & out["HomeTeam"].ne("") & out["AwayTeam"].ne("")].copy()
    out["source"] = "Football-Data"
    return out.sort_values(["Date", "league_code"]).reset_index(drop=True)


def download_league_season(league_code: str, start_year: int, timeout: int = 20) -> pd.DataFrame:
    url = football_data_url(league_code, start_year)
    raw = _download_bytes(url, timeout=timeout)
    frame = pd.read_csv(BytesIO(raw), on_bad_lines="skip")
    out = canonicalize_football_data(frame, league_code=league_code, season_start=start_year)
    out["source_url"] = url
    return out


def load_football_data(league_codes: Iterable[str], season_starts: Iterable[int],
                       timeout: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    frames, errors = [], []
    for league in league_codes:
        for season in season_starts:
            try:
                frames.append(download_league_season(league, int(season), timeout=timeout))
            except Exception as exc:
                errors.append(f"{league} {season_label(int(season))}: {exc}")
    if not frames:
        raise RuntimeError("No Football-Data files could be downloaded. " + (" | ".join(errors[:5]) if errors else ""))
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(["league_code", "Date"]).reset_index(drop=True), errors


def load_uploaded_csvs(uploaded_files) -> pd.DataFrame:
    frames = []
    for f in uploaded_files:
        frame = canonicalize_football_data(pd.read_csv(f, on_bad_lines="skip"))
        frame["source"] = getattr(f, "name", "uploaded_csv")
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV files supplied.")
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(["league_code", "Date"]).reset_index(drop=True)


def download_fixtures(timeout: int = 20) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(_download_bytes(FIXTURES_URL, timeout=timeout)), on_bad_lines="skip")
    if not {"Date", "HomeTeam", "AwayTeam"}.issubset(df.columns):
        raise ValueError("The Football-Data fixtures file does not contain the expected columns.")
    df["Date"] = _parse_date_series(df["Date"])
    if "Time" not in df.columns:
        df["Time"] = ""
    if "Div" not in df.columns:
        df["Div"] = ""
    return df.sort_values(["Date", "Div", "HomeTeam"]).reset_index(drop=True)


def completed_matches(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[raw["FTHG"].notna() & raw["FTAG"].notna()].copy()


@dataclass
class LeagueState:
    histories: Dict[str, List[dict]]
    elo: Dict[str, float]
    last_date: Dict[str, pd.Timestamp]
    matches_seen: int = 0
    home_goals: float = 0.0
    away_goals: float = 0.0
    draws: int = 0
    btts: int = 0
    o25: int = 0
    corners_sum: float = 0.0
    corners_n: int = 0


def _new_state() -> LeagueState:
    return LeagueState(defaultdict(list), defaultdict(lambda: 1500.0), {})


def _safe_mean(values, fallback=np.nan) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else float(fallback)


def _history_summary(history: List[dict], n: int, venue: Optional[str] = None) -> dict:
    rows = [r for r in history if not venue or r.get("venue") == venue][-n:]
    keys = ["ppg", "adj_ppg", "wppg", "gf", "ga", "gd", "shots", "sot", "corners",
            "opp_shots", "opp_sot", "opp_corners", "btts", "o25"]
    if not rows:
        return {k: np.nan for k in keys}
    points = [r["points"] for r in rows]
    weights = np.geomspace(0.55, 1.0, num=len(rows)) if len(rows) > 1 else np.array([1.0])
    gf, ga = _safe_mean([r["gf"] for r in rows]), _safe_mean([r["ga"] for r in rows])
    return {
        "ppg": _safe_mean(points),
        "adj_ppg": _safe_mean([r["points"] * (r.get("opp_elo", 1500.0) / 1500.0) for r in rows]),
        "wppg": float(np.average(points, weights=weights)),
        "gf": gf, "ga": ga, "gd": gf - ga,
        "shots": _safe_mean([r.get("shots", np.nan) for r in rows]),
        "sot": _safe_mean([r.get("sot", np.nan) for r in rows]),
        "corners": _safe_mean([r.get("corners", np.nan) for r in rows]),
        "opp_shots": _safe_mean([r.get("opp_shots", np.nan) for r in rows]),
        "opp_sot": _safe_mean([r.get("opp_sot", np.nan) for r in rows]),
        "opp_corners": _safe_mean([r.get("opp_corners", np.nan) for r in rows]),
        "btts": _safe_mean([r.get("btts", np.nan) for r in rows]),
        "o25": _safe_mean([r.get("o25", np.nan) for r in rows]),
    }


def _league_context(state: LeagueState) -> dict:
    n = state.matches_seen
    return {
        "league_home_goals": state.home_goals / n if n else 1.45,
        "league_away_goals": state.away_goals / n if n else 1.15,
        "league_draw_rate": state.draws / n if n else 0.26,
        "league_btts_rate": state.btts / n if n else 0.50,
        "league_o25_rate": state.o25 / n if n else 0.50,
        "league_corners_avg": state.corners_sum / state.corners_n if state.corners_n else 10.0,
    }


def _ratio(value, denom, fallback=1.0):
    if pd.isna(value) or pd.isna(denom) or abs(denom) < 1e-9:
        return fallback
    return float(np.clip(value / denom, 0.25, 3.0))


def _rest_days(state, team, match_date):
    last = state.last_date.get(team)
    return 7.0 if last is None or pd.isna(last) else float(np.clip((match_date - last).days, 1, 30))


def _elo_expectancy(home_elo, away_elo, home_advantage=65.0):
    return float(1.0 / (1.0 + 10.0 ** (-(home_elo + home_advantage - away_elo) / 400.0)))


def _feature_row(state: LeagueState, home: str, away: str, match_date: pd.Timestamp) -> dict:
    hh, ah = state.histories[home], state.histories[away]
    h5, a5 = _history_summary(hh, 5), _history_summary(ah, 5)
    h10, a10 = _history_summary(hh, 10), _history_summary(ah, 10)
    hv5, av5 = _history_summary(hh, 5, "H"), _history_summary(ah, 5, "A")
    hv10, av10 = _history_summary(hh, 10, "H"), _history_summary(ah, 10, "A")
    ctx = _league_context(state)
    helo, aelo = float(state.elo[home]), float(state.elo[away])
    home_attack = _ratio(hv10["gf"], ctx["league_home_goals"])
    away_attack = _ratio(av10["gf"], ctx["league_away_goals"])
    home_def_weak = _ratio(hv10["ga"], ctx["league_away_goals"])
    away_def_weak = _ratio(av10["ga"], ctx["league_home_goals"])
    lambda_h = float(np.clip(ctx["league_home_goals"] * home_attack * away_def_weak, 0.15, 4.5))
    lambda_a = float(np.clip(ctx["league_away_goals"] * away_attack * home_def_weak, 0.15, 4.5))
    diff = lambda x, y: x - y if pd.notna(x) and pd.notna(y) else np.nan
    return {
        "home_matches": len(hh), "away_matches": len(ah),
        "home_ppg_l5": h5["ppg"], "away_ppg_l5": a5["ppg"], "home_ppg_l10": h10["ppg"], "away_ppg_l10": a10["ppg"],
        "home_adj_ppg_l5": h5["adj_ppg"], "away_adj_ppg_l5": a5["adj_ppg"], "home_wppg_l5": h5["wppg"], "away_wppg_l5": a5["wppg"],
        "home_gf_l5": h5["gf"], "away_gf_l5": a5["gf"], "home_ga_l5": h5["ga"], "away_ga_l5": a5["ga"],
        "home_gf_l10": h10["gf"], "away_gf_l10": a10["gf"], "home_ga_l10": h10["ga"], "away_ga_l10": a10["ga"],
        "home_gd_l10": h10["gd"], "away_gd_l10": a10["gd"], "home_btts_l10": h10["btts"], "away_btts_l10": a10["btts"],
        "home_o25_l10": h10["o25"], "away_o25_l10": a10["o25"], "home_shots_l5": h5["shots"], "away_shots_l5": a5["shots"],
        "home_sot_l5": h5["sot"], "away_sot_l5": a5["sot"], "home_corners_l5": h5["corners"], "away_corners_l5": a5["corners"],
        "home_opp_shots_l5": h5["opp_shots"], "away_opp_shots_l5": a5["opp_shots"], "home_opp_sot_l5": h5["opp_sot"], "away_opp_sot_l5": a5["opp_sot"],
        "home_opp_corners_l5": h5["opp_corners"], "away_opp_corners_l5": a5["opp_corners"],
        "home_venue_ppg_l5": hv5["ppg"], "away_venue_ppg_l5": av5["ppg"], "home_venue_ppg_l10": hv10["ppg"], "away_venue_ppg_l10": av10["ppg"],
        "home_venue_gf_l10": hv10["gf"], "away_venue_gf_l10": av10["gf"], "home_venue_ga_l10": hv10["ga"], "away_venue_ga_l10": av10["ga"],
        "home_venue_sot_l5": hv5["sot"], "away_venue_sot_l5": av5["sot"], "home_venue_corners_l5": hv5["corners"], "away_venue_corners_l5": av5["corners"],
        "elo_home": helo, "elo_away": aelo, "elo_diff": helo - aelo, "elo_home_expectancy": _elo_expectancy(helo, aelo),
        "home_rest_days": _rest_days(state, home, match_date), "away_rest_days": _rest_days(state, away, match_date), **ctx,
        "home_attack_strength": home_attack, "away_attack_strength": away_attack, "home_def_weakness": home_def_weak, "away_def_weakness": away_def_weak,
        "lambda_form_home": lambda_h, "lambda_form_away": lambda_a,
        "ppg_diff_l5": diff(h5["ppg"], a5["ppg"]), "ppg_diff_l10": diff(h10["ppg"], a10["ppg"]),
        "gf_diff_l10": diff(h10["gf"], a10["gf"]), "ga_diff_l10": diff(h10["ga"], a10["ga"]), "gd_diff_l10": diff(h10["gd"], a10["gd"]),
        "sot_diff_l5": diff(h5["sot"], a5["sot"]), "corners_diff_l5": diff(h5["corners"], a5["corners"]),
        "venue_ppg_diff_l10": diff(hv10["ppg"], av10["ppg"]), "attack_strength_diff": home_attack - away_attack,
        "def_weakness_diff": home_def_weak - away_def_weak,
    }


def _append_history(state, row, home_pre_elo, away_pre_elo):
    hg, ag, home, away = float(row["FTHG"]), float(row["FTAG"]), row["HomeTeam"], row["AwayTeam"]
    common = {"btts": float(hg > 0 and ag > 0), "o25": float(hg + ag > 2.5)}
    state.histories[home].append({"date": row["Date"], "venue": "H", "opponent": away, "gf": hg, "ga": ag,
        "points": 3 if hg > ag else 1 if hg == ag else 0, "opp_elo": away_pre_elo,
        "shots": row.get("HS", np.nan), "sot": row.get("HST", np.nan), "corners": row.get("HC", np.nan),
        "opp_shots": row.get("AS", np.nan), "opp_sot": row.get("AST", np.nan), "opp_corners": row.get("AC", np.nan), **common})
    state.histories[away].append({"date": row["Date"], "venue": "A", "opponent": home, "gf": ag, "ga": hg,
        "points": 3 if ag > hg else 1 if hg == ag else 0, "opp_elo": home_pre_elo,
        "shots": row.get("AS", np.nan), "sot": row.get("AST", np.nan), "corners": row.get("AC", np.nan),
        "opp_shots": row.get("HS", np.nan), "opp_sot": row.get("HST", np.nan), "opp_corners": row.get("HC", np.nan), **common})
    state.last_date[home], state.last_date[away] = row["Date"], row["Date"]


def _update_state(state, row):
    home, away, hg, ag = row["HomeTeam"], row["AwayTeam"], float(row["FTHG"]), float(row["FTAG"])
    helo, aelo = float(state.elo[home]), float(state.elo[away])
    _append_history(state, row, helo, aelo)
    exp_h = _elo_expectancy(helo, aelo)
    score_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    delta = 24.0 * (score_h - exp_h)
    state.elo[home], state.elo[away] = helo + delta, aelo - delta
    state.matches_seen += 1
    state.home_goals += hg; state.away_goals += ag
    state.draws += int(hg == ag); state.btts += int(hg > 0 and ag > 0); state.o25 += int(hg + ag > 2.5)
    hc, ac = row.get("HC", np.nan), row.get("AC", np.nan)
    if pd.notna(hc) and pd.notna(ac):
        state.corners_sum += float(hc + ac); state.corners_n += 1


def build_feature_dataset(raw: pd.DataFrame, min_history: int = 3) -> pd.DataFrame:
    raw = completed_matches(raw).sort_values(["league_code", "Date"]).reset_index(drop=True)
    output = []
    for league_code, group in raw.groupby("league_code", sort=False):
        state = _new_state()
        for _, match in group.sort_values("Date").iterrows():
            home, away = match["HomeTeam"], match["AwayTeam"]
            features = _feature_row(state, home, away, match["Date"])
            if len(state.histories[home]) >= min_history and len(state.histories[away]) >= min_history:
                hg, ag = int(match["FTHG"]), int(match["FTAG"])
                total_corners = match.get("HC", np.nan) + match.get("AC", np.nan) if pd.notna(match.get("HC", np.nan)) and pd.notna(match.get("AC", np.nan)) else np.nan
                record = {"Date": match["Date"], "league_code": league_code, "HomeTeam": home, "AwayTeam": away,
                    "FTHG": hg, "FTAG": ag, "result": "H" if hg > ag else "A" if ag > hg else "D",
                    "btts": int(hg > 0 and ag > 0), "over25": int(hg + ag > 2.5), "total_goals": hg + ag,
                    "total_corners": total_corners, **features}
                record["corners_over85"] = int(total_corners > 8.5) if pd.notna(total_corners) else np.nan
                output.append(record)
            _update_state(state, match)
    return pd.DataFrame(output).sort_values("Date").reset_index(drop=True)


def build_prediction_row(raw: pd.DataFrame, league_code: str, home: str, away: str,
                         as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    matches = completed_matches(raw)
    matches = matches[matches["league_code"].astype(str) == str(league_code)].sort_values("Date")
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        matches = matches[matches["Date"] < as_of]
    if matches.empty:
        raise ValueError(f"No completed matches found for {league_code}.")
    state = _new_state()
    for _, match in matches.iterrows():
        _update_state(state, match)
    pred_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(matches["Date"].max()) + pd.Timedelta(days=7)
    return pd.DataFrame([{"Date": pred_date, "league_code": league_code, "HomeTeam": home, "AwayTeam": away,
                          **_feature_row(state, home, away, pred_date)}])


def available_teams(raw: pd.DataFrame, league_code: str) -> List[str]:
    df = raw[raw["league_code"].astype(str) == str(league_code)]
    teams = set(df["HomeTeam"].dropna().astype(str)) | set(df["AwayTeam"].dropna().astype(str))
    return sorted(t for t in teams if t and t.lower() != "nan")


def recent_team_matches(raw: pd.DataFrame, league_code: str, team: str, n: int = 10) -> pd.DataFrame:
    df = completed_matches(raw)
    df = df[(df["league_code"].astype(str) == str(league_code)) & ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))]
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
    return df.sort_values("Date", ascending=False)[cols].head(n).reset_index(drop=True)
