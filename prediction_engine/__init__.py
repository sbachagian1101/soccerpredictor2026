from __future__ import annotations

"""Compatibility layer for the Streamlit runtime.

Pandas Copy-on-Write can return read-only NumPy views from ``Series.to_numpy``.
The original corner bootstrap normalised such a view in place, which raises
``ValueError: output array is read-only`` on newer runtimes.  This package
loads the original engine and replaces only that routine with a copy-safe
implementation, keeping the public API unchanged.
"""

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

_ENGINE_PATH = Path(__file__).resolve().parent.parent / "prediction_engine.py"
_SPEC = importlib.util.spec_from_file_location("_soccer_prediction_engine_legacy", _ENGINE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load prediction engine from {_ENGINE_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)


def _safe_corner_models(
    home_df: pd.DataFrame,
    away_df: pd.DataFrame,
    home: dict,
    away: dict,
    rng: np.random.Generator,
):
    hcf = _LEGACY._coalesce(home["corners_for"], 4.8)
    hca = _LEGACY._coalesce(home["corners_against"], 5.0)
    acf = _LEGACY._coalesce(away["corners_for"], 4.8)
    aca = _LEGACY._coalesce(away["corners_against"], 5.0)
    mu_h = _LEGACY._clip(0.56 * hcf + 0.44 * aca, 1.2, 9.5)
    mu_a = _LEGACY._clip(0.56 * acf + 0.44 * hca, 1.2, 9.5)
    mu = mu_h + mu_a

    rows = []
    rows.append(("Corners Poisson", float(1 - poisson.cdf(8, mu)), mu, 0.27))
    k = 5.0
    rows.append(("Corners Negative Binomial", float(1 - nbinom.cdf(8, k, k / (k + mu))), mu, 0.23))

    totals, weights = [], []
    for df in (home_df, away_df):
        for _, r in df[df.weight > 0].iterrows():
            if pd.notna(r.corners_for) and pd.notna(r.corners_against):
                totals.append(float(r.corners_for + r.corners_against))
                weights.append(float(r.weight))
    if totals:
        shape = 12.0 + np.dot(totals, weights)
        rate = 1.3 + np.sum(weights)
        gmu = shape / rate
        p_gp = 1 - nbinom.cdf(8, shape, rate / (rate + 1))
    else:
        gmu = mu
        p_gp = 1 - nbinom.cdf(8, k, k / (k + mu))
    rows.append(("Corners Gamma-Poisson", float(p_gp), float(gmu), 0.18))

    def sampled_totals(df: pd.DataFrame, n: int) -> np.ndarray:
        d = df[(df.weight > 0) & df.corners_for.notna() & df.corners_against.notna()]
        if d.empty:
            return np.full(n, mu, dtype=float)

        # copy=True is intentional: under pandas Copy-on-Write the zero-copy
        # view is read-only, so in-place normalisation is unsafe.
        p = d["weight"].to_numpy(dtype=float, copy=True)
        total_weight = float(np.sum(p))
        if not np.isfinite(total_weight) or total_weight <= 0:
            p = np.full(len(d), 1.0 / len(d), dtype=float)
        else:
            p = p / total_weight

        idx = rng.choice(len(d), n, p=p)
        corners_for = d.iloc[idx]["corners_for"].to_numpy(dtype=float, copy=True)
        corners_against = d.iloc[idx]["corners_against"].to_numpy(dtype=float, copy=True)
        return corners_for + corners_against

    bt = 0.5 * sampled_totals(home_df, 15000) + 0.5 * sampled_totals(away_df, 15000)
    rows.append(("Corners Bootstrap", float(np.mean(bt >= 9)), float(np.mean(bt)), 0.17))

    pressure = (
        0.035 * ((_LEGACY._coalesce(home["shots_for"], 12) + _LEGACY._coalesce(away["shots_for"], 12)) - 24)
        + 0.010 * ((_LEGACY._coalesce(home["possession"], 50) + _LEGACY._coalesce(away["possession"], 50)) - 100)
    )
    pmu = _LEGACY._clip(mu * (1 + pressure), 4.0, 15.0)
    rows.append(("Pressure-adjusted corners", float(1 - poisson.cdf(8, pmu)), pmu, 0.15))

    z = sum(r[3] for r in rows)
    over = sum(r[1] * r[3] for r in rows) / z
    expected = sum(r[2] * r[3] for r in rows) / z
    table = pd.DataFrame(rows, columns=["Model", "P_Over_8_5", "Expected_Corners", "Weight"])
    return {"over85": over, "under85": 1 - over, "expected": expected, "breakdown": table}


# Patch the original module's global lookup. build_prediction therefore uses
# the safe implementation without changing the rest of the tested engine.
_LEGACY._corner_models = _safe_corner_models

build_prediction = _LEGACY.build_prediction
feature_comparison = _LEGACY.feature_comparison
score_matrix_dataframe = _LEGACY.score_matrix_dataframe
team_features = _LEGACY.team_features

__all__ = [
    "build_prediction",
    "feature_comparison",
    "score_matrix_dataframe",
    "team_features",
]
