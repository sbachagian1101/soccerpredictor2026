from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

WEEKDAYS = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
HEADER_RE = re.compile(
    rf"(?P<date>{WEEKDAYS}\s+{MONTHS}\s+\d{{1,2}},\s+\d{{4}})\s*-\s*"
    rf"(?P<time>\d{{1,2}}:\d{{2}}(?:am|pm)?)\s*\([^)]*?time\)\s*"
    rf"(?P<home>.+?)\s+vs\s+(?P<away>.+?)\s+Stats,\s*H2H\s*&\s*xG",
    re.IGNORECASE,
)


def _space(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", " ")
    text = text.replace("\\.", ".").replace("\\-", "-").replace("\\*", "*")
    return re.sub(r"\s+", " ", text).strip()


def _parse_float_pair(text: str, label: str, percent: bool = False) -> tuple[float, float]:
    unit = r"%?" if percent else ""
    token = r"(?:-?\d+(?:\.\d+)?|N/?A|-)"
    m = re.search(rf"{re.escape(label)}\s*({token}){unit}\s*({token}){unit}", text, re.IGNORECASE)
    if not m:
        return (np.nan, np.nan)

    def conv(v: str) -> float:
        if v.upper().replace("/", "") == "NA" or v == "-":
            return np.nan
        return float(v)

    return conv(m.group(1)), conv(m.group(2))


def _competition_from_prefix(prefix: str) -> tuple[str, str]:
    candidates = re.findall(
        r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,60})\s*/\s*([A-Za-z0-9À-ÿ][A-Za-z0-9À-ÿ .&'()-]{1,80})\s+Past H2H",
        prefix,
        re.IGNORECASE,
    )
    if not candidates:
        return "", ""
    country, competition = candidates[-1]
    country = re.sub(r".*?Dark\s+", "", country, flags=re.IGNORECASE).strip()
    return country.strip(), competition.strip()


def classify_match(competition: str) -> tuple[str, float]:
    c = (competition or "").lower()
    if any(k in c for k in ("friendly", "friendlies", "pre-season", "preseason")):
        return "Friendly", 0.40
    if any(k in c for k in ("playoff", "play-off", "relegation", "promotion", "final")):
        return "High importance", 1.12
    if any(k in c for k in ("champions league", "europa", "conference league", "libertadores", "sudamericana")):
        return "Continental", 1.08
    if any(k in c for k in ("cup", "pokal", "copa", "coupe", "fa trophy", "knockout")):
        return "Cup", 0.92
    if any(k in c for k in ("qualif", "qualification")):
        return "Qualifier", 1.02
    if competition:
        return "League", 1.00
    return "Competitive", 0.95


@dataclass
class MatchRecord:
    date: pd.Timestamp
    kickoff: str
    country: str
    competition: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    ht_home_goals: float
    ht_away_goals: float
    home_xg: float
    away_xg: float
    home_shots: float
    away_shots: float
    home_sot: float
    away_sot: float
    home_possession: float
    away_possession: float
    home_corners: float
    away_corners: float
    home_cards: float
    away_cards: float
    home_fouls: float
    away_fouls: float
    home_offsides: float
    away_offsides: float
    match_type: str
    auto_importance: float
    parse_quality: float


def parse_match_pages(raw_text: str) -> pd.DataFrame:
    """Parse copied FootyStats pages into one row per actual completed match.

    Only the fixture header, Final Results and the first post-match Data block are used.
    Prediction Stats, Current Form and Odds Market sections are deliberately ignored.
    """
    text = _space(raw_text)
    headers = list(HEADER_RE.finditer(text))
    records: list[MatchRecord] = []

    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        segment = text[h.end():end]
        prefix = text[max(0, start - 2000):start]

        score_m = re.search(r"Final Results\s*(\d+)\s*-\s*(\d+)", segment, re.IGNORECASE)
        if not score_m:
            continue
        hg, ag = int(score_m.group(1)), int(score_m.group(2))

        ht_m = re.search(r"HT\s*\(?\s*(\d+)\s*-\s*(\d+)\s*\)?", segment, re.IGNORECASE)
        hth, hta = (float(ht_m.group(1)), float(ht_m.group(2))) if ht_m else (np.nan, np.nan)

        final_pos = score_m.end()
        h2h_pos = segment.find("Head to Head Statistics", final_pos)
        stats_scope = segment[final_pos:h2h_pos if h2h_pos >= 0 else len(segment)]
        poss_pos = stats_scope.find("Possession")
        if poss_pos >= 0:
            stats_scope = stats_scope[poss_pos:]

        hp, ap = _parse_float_pair(stats_scope, "Possession", percent=True)
        hs, a_s = _parse_float_pair(stats_scope, "Shots")
        hsot, asot = _parse_float_pair(stats_scope, "Shots On Target")
        hcards, acards = _parse_float_pair(stats_scope, "Cards")
        hcorn, acorn = _parse_float_pair(stats_scope, "Corners")
        hfoul, afoul = _parse_float_pair(stats_scope, "Fouls")
        hoff, aoff = _parse_float_pair(stats_scope, "Offsides")
        hxg, axg = _parse_float_pair(stats_scope, "xG")

        country, competition = _competition_from_prefix(prefix)
        mtype, imp = classify_match(competition)
        available = [hxg, axg, hs, a_s, hsot, asot, hp, ap, hcorn, acorn]
        quality = float(np.mean([not pd.isna(x) for x in available])) if available else 0.0
        date = pd.to_datetime(h.group("date"), format="%A %b %d, %Y", errors="coerce")

        records.append(MatchRecord(
            date=date, kickoff=h.group("time"), country=country, competition=competition,
            home_team=h.group("home").strip(), away_team=h.group("away").strip(),
            home_goals=hg, away_goals=ag, ht_home_goals=hth, ht_away_goals=hta,
            home_xg=hxg, away_xg=axg, home_shots=hs, away_shots=a_s,
            home_sot=hsot, away_sot=asot, home_possession=hp, away_possession=ap,
            home_corners=hcorn, away_corners=acorn, home_cards=hcards, away_cards=acards,
            home_fouls=hfoul, away_fouls=afoul, home_offsides=hoff, away_offsides=aoff,
            match_type=mtype, auto_importance=imp, parse_quality=quality,
        ))

    if not records:
        return pd.DataFrame()
    out = pd.DataFrame([asdict(r) for r in records])
    out = out.drop_duplicates(subset=["date", "home_team", "away_team", "home_goals", "away_goals"], keep="first")
    return out.sort_values("date", ascending=False).reset_index(drop=True)


def detect_focus_team(matches: pd.DataFrame) -> str:
    if matches.empty:
        return ""
    counts = Counter(matches["home_team"].tolist() + matches["away_team"].tolist())
    return counts.most_common(1)[0][0] if counts else ""


def to_team_perspective(matches: pd.DataFrame, team: Optional[str] = None, max_matches: int = 5) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    team = team or detect_focus_team(matches)
    filt = matches[(matches.home_team == team) | (matches.away_team == team)].copy()
    filt = filt.sort_values("date", ascending=False).head(max_matches)
    rows = []
    for _, r in filt.iterrows():
        is_home = r.home_team == team
        gf, ga = (r.home_goals, r.away_goals) if is_home else (r.away_goals, r.home_goals)
        xg, xga = (r.home_xg, r.away_xg) if is_home else (r.away_xg, r.home_xg)
        shots_for, shots_against = (r.home_shots, r.away_shots) if is_home else (r.away_shots, r.home_shots)
        sot_for, sot_against = (r.home_sot, r.away_sot) if is_home else (r.away_sot, r.home_sot)
        poss_for = r.home_possession if is_home else r.away_possession
        corners_for, corners_against = (r.home_corners, r.away_corners) if is_home else (r.away_corners, r.home_corners)
        cards_for, cards_against = (r.home_cards, r.away_cards) if is_home else (r.away_cards, r.home_cards)
        fouls_for, fouls_against = (r.home_fouls, r.away_fouls) if is_home else (r.away_fouls, r.home_fouls)
        offsides_for, offsides_against = (r.home_offsides, r.away_offsides) if is_home else (r.away_offsides, r.home_offsides)
        rows.append({
            "include": True, "date": r.date, "team": team, "venue": "Home" if is_home else "Away",
            "opponent": r.away_team if is_home else r.home_team, "competition": r.competition,
            "match_type": r.match_type, "importance": float(r.auto_importance), "opponent_quality": 1.00,
            "result": "W" if gf > ga else "D" if gf == ga else "L", "gf": int(gf), "ga": int(ga),
            "xg": xg, "xga": xga, "shots_for": shots_for, "shots_against": shots_against,
            "sot_for": sot_for, "sot_against": sot_against, "possession": poss_for,
            "corners_for": corners_for, "corners_against": corners_against,
            "cards_for": cards_for, "cards_against": cards_against,
            "fouls_for": fouls_for, "fouls_against": fouls_against,
            "offsides_for": offsides_for, "offsides_against": offsides_against,
            "parse_quality": float(r.parse_quality),
        })
    return pd.DataFrame(rows)


def apply_observation_weights(df: pd.DataFrame, upcoming_date, relevant_venue: str, half_life_days: float = 45.0) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    upcoming = pd.Timestamp(upcoming_date)
    days = (upcoming - pd.to_datetime(out["date"])).dt.days.clip(lower=0)
    out["days_ago"] = days
    out["recency_weight"] = np.exp(-math.log(2.0) * days / max(half_life_days, 1.0))
    out["venue_weight"] = np.where(out["venue"].eq(relevant_venue), 1.08, 0.96)
    out["data_weight"] = 0.75 + 0.25 * out["parse_quality"].fillna(0.0).clip(0, 1)
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce").fillna(1.0).clip(0.2, 1.4)
    out["opponent_quality"] = pd.to_numeric(out["opponent_quality"], errors="coerce").fillna(1.0).clip(0.7, 1.3)
    out["weight"] = out["recency_weight"] * out["venue_weight"] * out["data_weight"] * out["importance"] * out["opponent_quality"]
    out.loc[~out["include"].astype(bool), "weight"] = 0.0
    return out


def parsing_diagnostics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"matches": 0, "xg": 0, "shots": 0, "corners": 0, "quality": 0.0}
    return {
        "matches": len(df),
        "xg": int(df["xg"].notna().sum()) if "xg" in df else 0,
        "shots": int(df["shots_for"].notna().sum()) if "shots_for" in df else 0,
        "corners": int(df["corners_for"].notna().sum()) if "corners_for" in df else 0,
        "quality": float(df["parse_quality"].mean()) if "parse_quality" in df else 0.0,
    }
