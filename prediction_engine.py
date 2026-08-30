from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom, poisson, skellam

MAX_GOALS = 7
EPS = 1e-9


def _clip(v, lo, hi):
    return float(np.clip(v, lo, hi))


def weighted_mean(df: pd.DataFrame, col: str, default=np.nan) -> float:
    if df.empty or col not in df:
        return float(default)
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df.get("weight", pd.Series(np.ones(len(df)))), errors="coerce").fillna(0).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float(default)
    return float(np.average(x[mask], weights=w[mask]))


def weighted_rate(df: pd.DataFrame, predicate) -> float:
    if df.empty:
        return 0.0
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0).to_numpy(dtype=float)
    y = np.asarray(predicate(df), dtype=float)
    mask = np.isfinite(w) & (w > 0) & np.isfinite(y)
    return float(np.average(y[mask], weights=w[mask])) if mask.any() else 0.0


def _coalesce(*vals, default=0.0):
    for v in vals:
        if v is not None and np.isfinite(v):
            return float(v)
    return float(default)


def _score_high(v: float, midpoint: float, scale: float) -> float:
    if not np.isfinite(v):
        return 50.0
    return _clip(50.0 + 48.0 * np.tanh((v - midpoint) / max(scale, 1e-6)), 1, 99)


def _score_low(v: float, midpoint: float, scale: float) -> float:
    if not np.isfinite(v):
        return 50.0
    return _clip(50.0 + 48.0 * np.tanh((midpoint - v) / max(scale, 1e-6)), 1, 99)


def team_features(df: pd.DataFrame) -> dict:
    active = df[df["weight"] > 0].copy()
    if active.empty:
        raise ValueError("No included matches have positive weight.")

    f = {}
    for col in [
        "gf", "ga", "xg", "xga", "shots_for", "shots_against", "sot_for", "sot_against",
        "possession", "corners_for", "corners_against", "cards_for", "cards_against",
        "fouls_for", "fouls_against", "offsides_for", "offsides_against",
    ]:
        f[col] = weighted_mean(active, col)

    f["ppg"] = weighted_rate(active, lambda d: np.where(d.result.eq("W"), 3.0, np.where(d.result.eq("D"), 1.0, 0.0)))
    f["win_rate"] = weighted_rate(active, lambda d: d.result.eq("W").astype(float))
    f["draw_rate"] = weighted_rate(active, lambda d: d.result.eq("D").astype(float))
    f["clean_sheet"] = weighted_rate(active, lambda d: d.ga.eq(0).astype(float))
    f["failed_to_score"] = weighted_rate(active, lambda d: d.gf.eq(0).astype(float))
    f["btts"] = weighted_rate(active, lambda d: ((d.gf > 0) & (d.ga > 0)).astype(float))
    f["over25"] = weighted_rate(active, lambda d: ((d.gf + d.ga) >= 3).astype(float))
    f["concede2plus"] = weighted_rate(active, lambda d: (d.ga >= 2).astype(float))
    f["xgd"] = _coalesce(f["xg"], f["gf"]) - _coalesce(f["xga"], f["ga"])
    f["gd"] = f["gf"] - f["ga"]
    f["finishing_delta"] = f["gf"] - _coalesce(f["xg"], f["gf"])
    f["defensive_delta"] = f["ga"] - _coalesce(f["xga"], f["ga"])
    f["conversion"] = f["gf"] / max(_coalesce(f["shots_for"], 10), 1.0)
    f["sot_rate"] = _coalesce(f["sot_for"], 0) / max(_coalesce(f["shots_for"], 10), 1.0)
    f["n_matches"] = int(len(active))
    f["effective_matches"] = float(active["weight"].sum() / max(active["weight"].max(), EPS))
    f["data_quality"] = weighted_mean(active, "parse_quality", default=0.0)

    recent = active.sort_values("date", ascending=False)
    if len(recent) >= 4:
        r2 = recent.head(2)
        old = recent.iloc[2:]
        recent_xgd = weighted_mean(r2, "xg", default=weighted_mean(r2, "gf")) - weighted_mean(r2, "xga", default=weighted_mean(r2, "ga"))
        old_xgd = weighted_mean(old, "xg", default=weighted_mean(old, "gf")) - weighted_mean(old, "xga", default=weighted_mean(old, "ga"))
        f["momentum"] = _clip(recent_xgd - old_xgd, -2.0, 2.0)
    else:
        f["momentum"] = 0.0

    attack_components = {
        "xG": _score_high(_coalesce(f["xg"], f["gf"]), 1.45, 0.75),
        "Goals": _score_high(f["gf"], 1.40, 0.95),
        "SOT": _score_high(_coalesce(f["sot_for"], 4.3), 4.3, 2.4),
        "Shots": _score_high(_coalesce(f["shots_for"], 12.0), 12.0, 6.0),
        "Corners": _score_high(_coalesce(f["corners_for"], 5.0), 5.0, 3.0),
        "Possession": _score_high(_coalesce(f["possession"], 50.0), 50.0, 18.0),
        "Conversion": _score_high(f["conversion"], 0.11, 0.07),
    }
    aw = {"xG": .28, "Goals": .18, "SOT": .17, "Shots": .10, "Corners": .08, "Possession": .06, "Conversion": .13}
    f["attack_strength"] = sum(attack_components[k] * aw[k] for k in aw)

    attack_weak_components = {
        "Low xG": _score_low(_coalesce(f["xg"], f["gf"]), 1.25, 0.65),
        "Low SOT": _score_low(_coalesce(f["sot_for"], 4.3), 3.8, 2.0),
        "Low conversion": _score_low(f["conversion"], 0.10, 0.06),
        "FTS": 100 * f["failed_to_score"],
        "Negative finishing": _score_high(max(-f["finishing_delta"], 0.0), 0.10, 0.45),
    }
    aww = {"Low xG": .28, "Low SOT": .20, "Low conversion": .18, "FTS": .22, "Negative finishing": .12}
    f["attack_weakness"] = sum(attack_weak_components[k] * aww[k] for k in aww)

    defense_components = {
        "xGA": _score_low(_coalesce(f["xga"], f["ga"]), 1.45, 0.75),
        "Goals allowed": _score_low(f["ga"], 1.40, 0.95),
        "SOT allowed": _score_low(_coalesce(f["sot_against"], 4.3), 4.3, 2.4),
        "Shots allowed": _score_low(_coalesce(f["shots_against"], 12.0), 12.0, 6.0),
        "Corners allowed": _score_low(_coalesce(f["corners_against"], 5.0), 5.0, 3.0),
        "Clean sheets": _clip(35 + 100 * f["clean_sheet"], 1, 99),
    }
    dw = {"xGA": .30, "Goals allowed": .18, "SOT allowed": .18, "Shots allowed": .11, "Corners allowed": .09, "Clean sheets": .14}
    f["defense_strength"] = sum(defense_components[k] * dw[k] for k in dw)

    defense_weak_components = {
        "High xGA": _score_high(_coalesce(f["xga"], f["ga"]), 1.55, 0.75),
        "High SOT allowed": _score_high(_coalesce(f["sot_against"], 4.3), 4.8, 2.4),
        "High shots allowed": _score_high(_coalesce(f["shots_against"], 12.0), 13.0, 6.0),
        "Concede 2+": 100 * f["concede2plus"],
        "Goals above xGA": _score_high(max(f["defensive_delta"], 0.0), 0.10, 0.50),
    }
    dww = {"High xGA": .30, "High SOT allowed": .20, "High shots allowed": .13, "Concede 2+": .22, "Goals above xGA": .15}
    f["defense_weakness"] = sum(defense_weak_components[k] * dww[k] for k in dww)

    f["attack_index"] = f["attack_strength"]
    f["defense_index"] = f["defense_strength"]
    f["attack_components"] = attack_components
    f["defense_components"] = defense_components
    return f


def poisson_matrix(lh: float, la: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    return m / m.sum()


def dixon_coles_matrix(lh: float, la: float, rho: float = -0.08, max_goals: int = MAX_GOALS) -> np.ndarray:
    m = poisson_matrix(lh, la, max_goals)
    tau = {(0, 0): 1 - lh * la * rho, (0, 1): 1 + lh * rho, (1, 0): 1 + la * rho, (1, 1): 1 - rho}
    for (i, j), t in tau.items():
        if i <= max_goals and j <= max_goals:
            m[i, j] *= max(t, 0.05)
    return m / m.sum()


def bivariate_poisson_matrix(lh: float, la: float, shared_frac: float = 0.10, max_goals: int = MAX_GOALS) -> np.ndarray:
    l3 = max(0.02, shared_frac * min(lh, la))
    l1, l2 = max(lh - l3, 0.02), max(la - l3, 0.02)
    m = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    base = math.exp(-(l1 + l2 + l3))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            s = 0.0
            for k in range(min(x, y) + 1):
                log_term = ((x-k) * math.log(l1) - gammaln(x-k+1) + (y-k) * math.log(l2) - gammaln(y-k+1) + k * math.log(l3) - gammaln(k+1))
                s += math.exp(log_term)
            m[x, y] = base * s
    return m / m.sum()


def negative_binomial_matrix(lh: float, la: float, dispersion: float = 3.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    ph = dispersion / (dispersion + lh)
    pa = dispersion / (dispersion + la)
    h = nbinom.pmf(goals, dispersion, ph)
    a = nbinom.pmf(goals, dispersion, pa)
    m = np.outer(h, a)
    return m / m.sum()


def matrix_probs(m: np.ndarray) -> tuple[float, float, float]:
    ph = float(np.tril(m, -1).sum())
    pd_ = float(np.trace(m))
    pa = float(np.triu(m, 1).sum())
    s = ph + pd_ + pa
    return ph / s, pd_ / s, pa / s


def _safe_lambdas(home: dict, away: dict, mode: str) -> tuple[float, float]:
    h_xg = _coalesce(home["xg"], home["gf"], default=1.35)
    a_xg = _coalesce(away["xg"], away["gf"], default=1.15)
    h_xga = _coalesce(home["xga"], home["ga"], default=1.25)
    a_xga = _coalesce(away["xga"], away["ga"], default=1.45)

    if mode == "goals":
        lh = 0.52 * home["gf"] + 0.48 * away["ga"]
        la = 0.52 * away["gf"] + 0.48 * home["ga"]
    elif mode == "xg":
        lh = 0.56 * h_xg + 0.44 * a_xga
        la = 0.56 * a_xg + 0.44 * h_xga
    elif mode == "strength":
        base_h = 0.55 * h_xg + 0.45 * a_xga
        base_a = 0.55 * a_xg + 0.45 * h_xga
        att_adj_h = 1 + (home["attack_strength"] - 50) / 250
        def_adj_a = 1 + (away["defense_weakness"] - 50) / 280
        att_adj_a = 1 + (away["attack_strength"] - 50) / 250
        def_adj_h = 1 + (home["defense_weakness"] - 50) / 280
        lh = base_h * att_adj_h * def_adj_a
        la = base_a * att_adj_a * def_adj_h
    else:
        lh = 0.45 * h_xg + 0.25 * home["gf"] + 0.20 * a_xga + 0.10 * away["ga"]
        la = 0.45 * a_xg + 0.25 * away["gf"] + 0.20 * h_xga + 0.10 * home["ga"]

    lh *= 1.06
    la *= 0.97
    return _clip(lh, 0.15, 3.8), _clip(la, 0.12, 3.6)


def _dirichlet_form_probs(home_df: pd.DataFrame, away_df: pd.DataFrame, home: dict, away: dict) -> tuple[float, float, float]:
    def pseudo(d):
        w = d["weight"].to_numpy(float)
        return np.array([w[d.result.eq("W").to_numpy()].sum(), w[d.result.eq("D").to_numpy()].sum(), w[d.result.eq("L").to_numpy()].sum()])
    hp = pseudo(home_df) + np.array([1.15, 1.0, 0.95])
    ap = pseudo(away_df) + np.array([0.95, 1.0, 1.10])
    hform = hp / hp.sum()
    aform = ap / ap.sum()
    ph = math.sqrt(hform[0] * aform[2])
    pd_ = math.sqrt(hform[1] * aform[1])
    pa = math.sqrt(hform[2] * aform[0])
    edge = np.tanh((home["xgd"] - away["xgd"]) / 1.5)
    ph *= (1 + 0.18 * edge)
    pa *= (1 - 0.18 * edge)
    z = ph + pd_ + pa
    return ph/z, pd_/z, pa/z


def _elo_style_probs(home: dict, away: dict, lh: float, la: float) -> tuple[float, float, float]:
    diff = 65.0 + 115.0 * (home["ppg"] - away["ppg"]) + 125.0 * (home["xgd"] - away["xgd"])
    q = 1 / (1 + 10 ** (-diff / 400.0))
    total = lh + la
    draw = _clip(0.29 * math.exp(-abs(diff) / 650.0) * (1.10 if total < 2.4 else 0.93), 0.14, 0.34)
    return q * (1-draw), draw, (1-q) * (1-draw)


def _power_probs(home: dict, away: dict) -> tuple[float, float, float]:
    edge = ((home["attack_strength"] - away["defense_strength"]) + (home["defense_strength"] - away["attack_strength"]) + 0.30 * (away["defense_weakness"] - home["defense_weakness"]) + 5.0)
    q = 1 / (1 + math.exp(-edge / 16.0))
    draw = _clip(0.27 * math.exp(-abs(edge) / 55.0), 0.13, 0.31)
    return q*(1-draw), draw, (1-q)*(1-draw)


def _bootstrap_model(home_df: pd.DataFrame, away_df: pd.DataFrame, rng: np.random.Generator, sims: int = 25000):
    h = home_df[home_df.weight > 0].reset_index(drop=True)
    a = away_df[away_df.weight > 0].reset_index(drop=True)
    hp = h.weight.to_numpy(float); hp = hp / hp.sum()
    ap = a.weight.to_numpy(float); ap = ap / ap.sum()
    hi = rng.choice(len(h), size=sims, p=hp)
    ai = rng.choice(len(a), size=sims, p=ap)
    hs = h.iloc[hi]
    ass = a.iloc[ai]

    hxg = pd.to_numeric(hs.xg, errors="coerce").fillna(pd.to_numeric(hs.gf, errors="coerce")).to_numpy(float)
    axga = pd.to_numeric(ass.xga, errors="coerce").fillna(pd.to_numeric(ass.ga, errors="coerce")).to_numpy(float)
    axg = pd.to_numeric(ass.xg, errors="coerce").fillna(pd.to_numeric(ass.gf, errors="coerce")).to_numpy(float)
    hxga = pd.to_numeric(hs.xga, errors="coerce").fillna(pd.to_numeric(hs.ga, errors="coerce")).to_numpy(float)
    lam_h = np.clip((0.58*hxg + 0.42*axga) * 1.06, .12, 4.2)
    lam_a = np.clip((0.58*axg + 0.42*hxga) * .97, .10, 4.0)
    gh = rng.poisson(lam_h); ga = rng.poisson(lam_a)
    ph = float(np.mean(gh > ga)); pdraw = float(np.mean(gh == ga)); pa = 1-ph-pdraw
    m = np.zeros((MAX_GOALS+1, MAX_GOALS+1), float)
    for x, y in zip(np.minimum(gh, MAX_GOALS), np.minimum(ga, MAX_GOALS)):
        m[int(x), int(y)] += 1
    m /= m.sum()
    return (ph, pdraw, pa), m, float(np.mean(lam_h)), float(np.mean(lam_a))


def _gamma_poisson_model(home_df: pd.DataFrame, away_df: pd.DataFrame):
    def posterior_rate(df, scored=True):
        w = df.weight.to_numpy(float)
        vals = pd.to_numeric(df["gf" if scored else "ga"], errors="coerce").fillna(0).to_numpy(float)
        shape = 2.7 + np.sum(w * vals)
        rate = 2.0 + np.sum(w)
        return shape, rate
    h_att = posterior_rate(home_df, True); a_def = posterior_rate(away_df, False)
    a_att = posterior_rate(away_df, True); h_def = posterior_rate(home_df, False)
    lh = 0.54*(h_att[0]/h_att[1]) + 0.46*(a_def[0]/a_def[1])
    la = 0.54*(a_att[0]/a_att[1]) + 0.46*(h_def[0]/h_def[1])
    lh, la = _clip(lh*1.06, .15, 3.8), _clip(la*.97, .12, 3.6)
    kh = max(2.0, 0.5*(h_att[0]+a_def[0])); ka = max(2.0, 0.5*(a_att[0]+h_def[0]))
    goals = np.arange(MAX_GOALS+1)
    hh = nbinom.pmf(goals, kh, kh/(kh+lh))
    aa = nbinom.pmf(goals, ka, ka/(ka+la))
    m = np.outer(hh, aa); m /= m.sum()
    return (lh, la), m


def _corner_models(home_df: pd.DataFrame, away_df: pd.DataFrame, home: dict, away: dict, rng: np.random.Generator):
    hcf = _coalesce(home["corners_for"], 4.8); hca = _coalesce(home["corners_against"], 5.0)
    acf = _coalesce(away["corners_for"], 4.8); aca = _coalesce(away["corners_against"], 5.0)
    mu_h = _clip(0.56*hcf + 0.44*aca, 1.2, 9.5)
    mu_a = _clip(0.56*acf + 0.44*hca, 1.2, 9.5)
    mu = mu_h + mu_a

    rows = []
    rows.append(("Corners Poisson", float(1 - poisson.cdf(8, mu)), mu, .27))
    k = 5.0
    rows.append(("Corners Negative Binomial", float(1 - nbinom.cdf(8, k, k/(k+mu))), mu, .23))

    totals = []; weights = []
    for df in (home_df, away_df):
        for _, r in df[df.weight > 0].iterrows():
            if pd.notna(r.corners_for) and pd.notna(r.corners_against):
                totals.append(float(r.corners_for + r.corners_against)); weights.append(float(r.weight))
    if totals:
        shape = 12.0 + np.dot(totals, weights)
        rate = 1.3 + np.sum(weights)
        gmu = shape/rate
        p_gp = 1 - nbinom.cdf(8, shape, rate/(rate+1))
    else:
        gmu, p_gp = mu, 1 - nbinom.cdf(8, k, k/(k+mu))
    rows.append(("Corners Gamma-Poisson", float(p_gp), float(gmu), .18))

    def sampled_totals(df, n):
        d = df[(df.weight > 0) & df.corners_for.notna() & df.corners_against.notna()]
        if d.empty:
            return np.full(n, mu)
        p = d.weight.to_numpy(float); p /= p.sum()
        idx = rng.choice(len(d), n, p=p)
        return d.iloc[idx].corners_for.to_numpy(float) + d.iloc[idx].corners_against.to_numpy(float)
    bt = 0.5*sampled_totals(home_df, 15000) + 0.5*sampled_totals(away_df, 15000)
    rows.append(("Corners Bootstrap", float(np.mean(bt >= 9)), float(np.mean(bt)), .17))

    pressure = 0.035*((_coalesce(home["shots_for"], 12)+_coalesce(away["shots_for"], 12))-24) + 0.010*((_coalesce(home["possession"], 50)+_coalesce(away["possession"], 50))-100)
    pmu = _clip(mu*(1+pressure), 4.0, 15.0)
    rows.append(("Pressure-adjusted corners", float(1-poisson.cdf(8, pmu)), pmu, .15))

    z = sum(r[3] for r in rows)
    over = sum(r[1]*r[3] for r in rows)/z
    expected = sum(r[2]*r[3] for r in rows)/z
    table = pd.DataFrame(rows, columns=["Model", "P_Over_8_5", "Expected_Corners", "Weight"])
    return {"over85": over, "under85": 1-over, "expected": expected, "breakdown": table}


def build_prediction(home_df: pd.DataFrame, away_df: pd.DataFrame, seed: int = 42) -> dict:
    home = team_features(home_df); away = team_features(away_df)
    home_name = str(home_df.iloc[0].team); away_name = str(away_df.iloc[0].team)
    rng = np.random.default_rng(seed)

    models = []; matrices: Dict[str, np.ndarray] = {}; lambdas: Dict[str, tuple[float,float]] = {}

    def add(name, probs, weight, matrix=None, lam=None, family="Goal distribution"):
        ph,pd_,pa = [float(x) for x in probs]
        z = ph+pd_+pa
        models.append({"Model":name,"Home":ph/z,"Draw":pd_/z,"Away":pa/z,"BaseWeight":weight,"Family":family})
        if matrix is not None: matrices[name]=matrix
        if lam is not None: lambdas[name]=lam

    lam = _safe_lambdas(home, away, "goals"); m = poisson_matrix(*lam); add("Poisson (results)", matrix_probs(m), .075, m, lam)
    lam = _safe_lambdas(home, away, "xg"); m = poisson_matrix(*lam); add("xG Poisson", matrix_probs(m), .115, m, lam)
    lam = _safe_lambdas(home, away, "strength"); m = poisson_matrix(*lam); add("Attack-Defence Poisson", matrix_probs(m), .105, m, lam)
    lam = _safe_lambdas(home, away, "blend"); m = dixon_coles_matrix(*lam); add("Dixon-Coles", matrix_probs(m), .125, m, lam)
    lam = _safe_lambdas(home, away, "xg"); m = bivariate_poisson_matrix(*lam); add("Bivariate Poisson", matrix_probs(m), .095, m, lam)
    lam = _safe_lambdas(home, away, "blend"); m = negative_binomial_matrix(*lam, dispersion=2.8); add("Negative Binomial", matrix_probs(m), .075, m, lam)
    lam, m = _gamma_poisson_model(home_df, away_df); add("Bayesian Gamma-Poisson", matrix_probs(m), .085, m, lam, "Bayesian goal model")
    lam = _safe_lambdas(home, away, "blend")
    pd_s = float(skellam.pmf(0, *lam)); pa_s = float(skellam.cdf(-1, *lam)); ph_s = 1-pd_s-pa_s
    add("Skellam Difference", (ph_s,pd_s,pa_s), .055, None, lam, "Goal-difference")
    add("Elo-style Performance", _elo_style_probs(home, away, *lam), .065, None, None, "Rating")
    add("Power Index", _power_probs(home, away), .060, None, None, "Rating")
    add("Bayesian Form", _dirichlet_form_probs(home_df, away_df, home, away), .060, None, None, "Form posterior")
    bprobs, bm, blh, bla = _bootstrap_model(home_df, away_df, rng)
    add("Weighted Bootstrap", bprobs, .085, bm, (blh,bla), "Simulation")

    table = pd.DataFrame(models)
    xg_quality = min(
        weighted_rate(home_df[home_df.weight>0], lambda d: d.xg.notna().astype(float)),
        weighted_rate(away_df[away_df.weight>0], lambda d: d.xg.notna().astype(float)),
    )
    xg_models = {"xG Poisson", "Dixon-Coles", "Bivariate Poisson", "Attack-Defence Poisson"}
    table["QualityFactor"] = np.where(table.Model.isin(xg_models), 0.45 + 0.55*xg_quality, 1.0)
    table["Weight"] = table.BaseWeight * table.QualityFactor
    table["Weight"] /= table.Weight.sum()

    ph = float(np.sum(table.Home*table.Weight)); pd_ = float(np.sum(table.Draw*table.Weight)); pa = float(np.sum(table.Away*table.Weight))
    probs = np.array([ph,pd_,pa]); probs /= probs.sum(); ph,pd_,pa = probs.tolist()

    score_weights = {name: float(table.loc[table.Model.eq(name), "Weight"].iloc[0]) for name in matrices}
    sw = sum(score_weights.values())
    score_matrix = sum(matrices[name]*(w/sw) for name,w in score_weights.items())
    score_matrix /= score_matrix.sum()

    lw = {name: score_weights.get(name,0) for name in lambdas if name in score_weights}
    lz = sum(lw.values()) or 1.0
    exp_home = sum(lambdas[name][0]*w for name,w in lw.items())/lz
    exp_away = sum(lambdas[name][1]*w for name,w in lw.items())/lz

    btts_yes = float(score_matrix[1:,1:].sum())
    over25 = float(sum(score_matrix[i,j] for i in range(score_matrix.shape[0]) for j in range(score_matrix.shape[1]) if i+j >= 3))

    top_scores=[]
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            top_scores.append((f"{i}-{j}", float(score_matrix[i,j])))
    top_scores = sorted(top_scores, key=lambda x:x[1], reverse=True)[:8]
    corners = _corner_models(home_df, away_df, home, away, rng)

    model_arr = table[["Home","Draw","Away"]].to_numpy(float)
    dispersion = float(np.mean(np.std(model_arr, axis=0)))
    agreement = _clip(1 - dispersion/0.16, 0, 1)
    sorted_probs = sorted([ph,pd_,pa], reverse=True)
    separation = _clip((sorted_probs[0]-sorted_probs[1])/0.25, 0, 1)
    quality = 0.5*home["data_quality"] + 0.5*away["data_quality"]
    sample = min(1.0, (home["n_matches"]+away["n_matches"])/10.0)
    confidence = 100*_clip(0.38*quality + 0.32*agreement + 0.20*separation + 0.10*sample, 0, 1)

    labels = [home_name, "Draw", away_name]
    best_i = int(np.argmax([ph,pd_,pa]))
    return {
        "home_team": home_name, "away_team": away_name,
        "p_home": ph, "p_draw": pd_, "p_away": pa, "pick": labels[best_i],
        "expected_home_goals": exp_home, "expected_away_goals": exp_away,
        "btts_yes": btts_yes, "btts_no": 1-btts_yes, "over25": over25, "under25": 1-over25,
        "corners": corners, "score_matrix": score_matrix, "top_scores": top_scores,
        "models": table.sort_values("Weight", ascending=False).reset_index(drop=True),
        "home_features": home, "away_features": away, "confidence": confidence, "xg_quality": xg_quality,
    }


def score_matrix_dataframe(result: dict, max_display: int = 6) -> pd.DataFrame:
    m = result["score_matrix"][:max_display+1,:max_display+1]
    df = pd.DataFrame(m*100, index=[str(i) for i in range(max_display+1)], columns=[str(i) for i in range(max_display+1)])
    df.index.name = f"{result['home_team']} goals ↓ / {result['away_team']} goals →"
    return df


def feature_comparison(result: dict) -> pd.DataFrame:
    h, a = result["home_features"], result["away_features"]
    metrics = [
        ("Weighted xG", "xg"), ("Weighted xGA", "xga"), ("Goals", "gf"), ("Goals Against", "ga"),
        ("Shots", "shots_for"), ("Shots on Target", "sot_for"), ("Corners For", "corners_for"),
        ("Corners Against", "corners_against"), ("Possession %", "possession"), ("PPG", "ppg"),
        ("xG Difference", "xgd"), ("Finishing Δ (G-xG)", "finishing_delta"),
        ("Defensive Δ (GA-xGA)", "defensive_delta"), ("Momentum", "momentum"),
    ]
    return pd.DataFrame([{"Metric":label, result["home_team"]:h.get(key), result["away_team"]:a.get(key)} for label,key in metrics])
