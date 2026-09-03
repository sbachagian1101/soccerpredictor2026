from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

COMPLETED_STATUSES = {"complete", "completed", "finished", "ft", "full time"}

REQUIRED_COLUMNS = {
    "status",
    "home_team_name",
    "away_team_name",
    "home_team_goal_count",
    "away_team_goal_count",
}

PERFORMANCE_COLUMNS = [
    "team_a_xg", "team_b_xg",
    "home_team_shots", "away_team_shots",
    "home_team_shots_on_target", "away_team_shots_on_target",
    "home_team_corner_count", "away_team_corner_count",
    "home_team_possession", "away_team_possession",
    "home_team_yellow_cards", "away_team_yellow_cards",
    "home_team_red_cards", "away_team_red_cards",
    "home_team_fouls", "away_team_fouls",
]


def _clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.mask(out < 0, np.nan)


def _parse_dates(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        parsed = pd.to_datetime(ts, unit="s", errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed.dt.tz_localize(None)
    if "date_GMT" in df.columns:
        parsed = pd.to_datetime(df["date_GMT"], errors="coerce", utc=True)
        return parsed.dt.tz_localize(None)
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def canonicalize_footystats(df: pd.DataFrame, source_name: str = "uploaded.csv") -> pd.DataFrame:
    out = df.copy()
    missing = REQUIRED_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(
            f"{source_name} is not a recognised FootyStats match CSV. Missing columns: {sorted(missing)}"
        )

    out["source_file"] = source_name
    out["match_date"] = _parse_dates(out)
    out["status"] = out["status"].astype(str).str.lower().str.strip()
    out["home_team_name"] = out["home_team_name"].astype(str).str.strip()
    out["away_team_name"] = out["away_team_name"].astype(str).str.strip()

    numeric = [
        "home_team_goal_count", "away_team_goal_count",
        "home_team_goal_count_half_time", "away_team_goal_count_half_time",
        "total_goal_count", "total_goals_at_half_time",
    ] + PERFORMANCE_COLUMNS
    for col in numeric:
        if col in out.columns:
            out[col] = _clean_numeric(out[col])

    # FootyStats future fixtures often contain 0-0 placeholder values. Status is therefore mandatory.
    out["is_completed"] = out["status"].isin(COMPLETED_STATUSES)
    out = out[out["home_team_name"].ne("") & out["away_team_name"].ne("")].copy()
    return out.sort_values(["match_date", "home_team_name", "away_team_name"], na_position="last").reset_index(drop=True)


def load_footystats_uploads(uploaded_files: Iterable) -> Tuple[pd.DataFrame, list[str]]:
    frames = []
    notices: list[str] = []
    for f in uploaded_files:
        name = getattr(f, "name", "uploaded.csv")
        try:
            raw = pd.read_csv(f, on_bad_lines="skip")
            frames.append(canonicalize_footystats(raw, source_name=name))
        except Exception as exc:
            notices.append(f"{name}: {exc}")

    if not frames:
        raise ValueError("No valid FootyStats CSV files could be loaded.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_cols = [c for c in ["timestamp", "match_date", "home_team_name", "away_team_name"] if c in combined.columns]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
    removed = before - len(combined)
    if removed:
        notices.append(f"Removed {removed} duplicate match rows across uploaded files.")
    return combined, notices


def completed_matches(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[raw["is_completed"]].copy()


def home_team_options(raw: pd.DataFrame) -> list[str]:
    completed = completed_matches(raw)
    return sorted(completed["home_team_name"].dropna().astype(str).unique().tolist())


def away_team_options(raw: pd.DataFrame) -> list[str]:
    completed = completed_matches(raw)
    return sorted(completed["away_team_name"].dropna().astype(str).unique().tolist())


def _num(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
    return np.nan if pd.isna(v) or float(v) < 0 else float(v)


def _sum_optional(row: pd.Series, *cols: str) -> float:
    vals = [_num(row, c) for c in cols]
    finite = [v for v in vals if np.isfinite(v)]
    return float(sum(finite)) if finite else np.nan


def build_venue_team_data(raw: pd.DataFrame, team: str, venue: str) -> pd.DataFrame:
    """Return standardized observations for one team at one venue only.

    venue='H' uses only rows where the selected team is home.
    venue='A' uses only rows where the selected team is away.
    """
    venue = venue.upper()
    if venue not in {"H", "A"}:
        raise ValueError("venue must be 'H' or 'A'")

    completed = completed_matches(raw)
    if venue == "H":
        selected = completed[completed["home_team_name"].eq(team)].copy()
    else:
        selected = completed[completed["away_team_name"].eq(team)].copy()

    rows = []
    for _, r in selected.sort_values("match_date").iterrows():
        if venue == "H":
            gf, ga = _num(r, "home_team_goal_count"), _num(r, "away_team_goal_count")
            xg, xga = _num(r, "team_a_xg"), _num(r, "team_b_xg")
            shots_for, shots_against = _num(r, "home_team_shots"), _num(r, "away_team_shots")
            sot_for, sot_against = _num(r, "home_team_shots_on_target"), _num(r, "away_team_shots_on_target")
            corners_for, corners_against = _num(r, "home_team_corner_count"), _num(r, "away_team_corner_count")
            possession = _num(r, "home_team_possession")
            cards_for = _sum_optional(r, "home_team_yellow_cards", "home_team_red_cards")
            cards_against = _sum_optional(r, "away_team_yellow_cards", "away_team_red_cards")
            fouls_for, fouls_against = _num(r, "home_team_fouls"), _num(r, "away_team_fouls")
            opponent = str(r["away_team_name"])
        else:
            gf, ga = _num(r, "away_team_goal_count"), _num(r, "home_team_goal_count")
            xg, xga = _num(r, "team_b_xg"), _num(r, "team_a_xg")
            shots_for, shots_against = _num(r, "away_team_shots"), _num(r, "home_team_shots")
            sot_for, sot_against = _num(r, "away_team_shots_on_target"), _num(r, "home_team_shots_on_target")
            corners_for, corners_against = _num(r, "away_team_corner_count"), _num(r, "home_team_corner_count")
            possession = _num(r, "away_team_possession")
            cards_for = _sum_optional(r, "away_team_yellow_cards", "away_team_red_cards")
            cards_against = _sum_optional(r, "home_team_yellow_cards", "home_team_red_cards")
            fouls_for, fouls_against = _num(r, "away_team_fouls"), _num(r, "home_team_fouls")
            opponent = str(r["home_team_name"])

        if not np.isfinite(gf) or not np.isfinite(ga):
            continue
        result = "W" if gf > ga else "D" if gf == ga else "L"
        quality_values = [xg, xga, shots_for, shots_against, sot_for, sot_against,
                          corners_for, corners_against, possession, cards_for, cards_against,
                          fouls_for, fouls_against]
        parse_quality = float(np.mean([np.isfinite(v) for v in quality_values]))

        rows.append({
            "team": team,
            "venue": venue,
            "date": pd.Timestamp(r["match_date"]) if pd.notna(r["match_date"]) else pd.NaT,
            "source_file": r.get("source_file", ""),
            "opponent": opponent,
            "result": result,
            "gf": gf,
            "ga": ga,
            "xg": xg,
            "xga": xga,
            "shots_for": shots_for,
            "shots_against": shots_against,
            "sot_for": sot_for,
            "sot_against": sot_against,
            "possession": possession,
            "corners_for": corners_for,
            "corners_against": corners_against,
            "cards_for": cards_for,
            "cards_against": cards_against,
            "fouls_for": fouls_for,
            "fouls_against": fouls_against,
            "offsides_for": np.nan,
            "offsides_against": np.nan,
            "weight": 1.0,
            "parse_quality": parse_quality,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def venue_summary(team_df: pd.DataFrame) -> dict:
    if team_df.empty:
        return {}
    return {
        "matches": len(team_df),
        "wins": int((team_df["result"] == "W").sum()),
        "draws": int((team_df["result"] == "D").sum()),
        "losses": int((team_df["result"] == "L").sum()),
        "ppg": float(np.mean(np.where(team_df.result.eq("W"), 3, np.where(team_df.result.eq("D"), 1, 0)))),
        "gf": float(team_df["gf"].mean()),
        "ga": float(team_df["ga"].mean()),
        "xg": float(team_df["xg"].mean()) if team_df["xg"].notna().any() else np.nan,
        "xga": float(team_df["xga"].mean()) if team_df["xga"].notna().any() else np.nan,
        "shots": float(team_df["shots_for"].mean()) if team_df["shots_for"].notna().any() else np.nan,
        "sot": float(team_df["sot_for"].mean()) if team_df["sot_for"].notna().any() else np.nan,
        "corners": float(team_df["corners_for"].mean()) if team_df["corners_for"].notna().any() else np.nan,
    }
