from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import FEATURE_COLUMNS

RESULT_ORDER = ["H", "D", "A"]


def poisson_pmf(k: int, lam: float) -> float:
    lam = float(np.clip(lam, 0.05, 8.0))
    return exp(-lam) * (lam ** k) / factorial(k)


def score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 6,
                 dixon_coles_rho: Optional[float] = None) -> np.ndarray:
    lh, la = float(np.clip(lambda_home, 0.08, 6.0)), float(np.clip(lambda_away, 0.08, 6.0))
    m = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, lh) * poisson_pmf(a, la)
            if dixon_coles_rho is not None:
                rho = float(np.clip(dixon_coles_rho, -0.25, 0.25))
                if h == 0 and a == 0: p *= max(0.2, 1.0 - lh * la * rho)
                elif h == 0 and a == 1: p *= max(0.2, 1.0 + lh * rho)
                elif h == 1 and a == 0: p *= max(0.2, 1.0 + la * rho)
                elif h == 1 and a == 1: p *= max(0.2, 1.0 - rho)
            m[h, a] = max(p, 0.0)
    return m / m.sum()


def matrix_1x2(m: np.ndarray) -> np.ndarray:
    probs = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()], dtype=float)
    return probs / probs.sum()


def matrix_markets(m: np.ndarray) -> dict:
    home, draw, away = matrix_1x2(m)
    btts = float(m[1:, 1:].sum())
    over25 = sum(float(m[h, a]) for h in range(m.shape[0]) for a in range(m.shape[1]) if h + a >= 3)
    return {"H": home, "D": draw, "A": away, "BTTS": btts, "O25": over25}


def _class_probs(model, X):
    raw, classes = model.predict_proba(X)[0], list(model.classes_)
    return np.array([raw[classes.index(c)] if c in classes else 0.0 for c in RESULT_ORDER])


def _class_probs_many(model, X):
    raw, classes = model.predict_proba(X), list(model.classes_)
    out = np.column_stack([raw[:, classes.index(c)] if c in classes else np.zeros(len(X)) for c in RESULT_ORDER])
    return out / out.sum(axis=1, keepdims=True)


def _safe_logloss(y_true, probs):
    try:
        return float(log_loss(y_true, probs[:, [2, 1, 0]], labels=["A", "D", "H"]))
    except Exception:
        return 9.0


def _weight_from_logloss(loss):
    return float(np.clip(np.exp(-max(loss, 0.05)), 0.08, 1.0))


def _base_classifier_models(random_state=42):
    scaled = lambda estimator: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)])
    tree = lambda estimator: Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=1800, C=0.75)),
        "Random Forest": tree(RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=5, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)),
        "Extra Trees": tree(ExtraTreesClassifier(n_estimators=200, max_depth=9, min_samples_leaf=4, class_weight="balanced", random_state=random_state, n_jobs=-1)),
        "Gradient Boosting": tree(GradientBoostingClassifier(n_estimators=120, learning_rate=0.035, max_depth=2, random_state=random_state)),
        "Hist Gradient Boosting": tree(HistGradientBoostingClassifier(max_iter=160, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=1.0, random_state=random_state)),
        "Gaussian NB": scaled(GaussianNB(var_smoothing=0.08)),
    }


def _binary_models(random_state=42):
    scaled = lambda estimator: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)])
    tree = lambda estimator: Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    return {
        "Logistic": scaled(LogisticRegression(max_iter=1500, C=0.8)),
        "Random Forest": tree(RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=5, random_state=random_state, n_jobs=-1)),
        "Extra Trees": tree(ExtraTreesClassifier(n_estimators=170, max_depth=8, min_samples_leaf=4, random_state=random_state, n_jobs=-1)),
        "Gradient Boosting": tree(GradientBoostingClassifier(n_estimators=100, learning_rate=0.04, max_depth=2, random_state=random_state)),
        "Hist Gradient Boosting": tree(HistGradientBoostingClassifier(max_iter=140, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=random_state)),
    }


def _goal_regressors(random_state=42):
    linear = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", PoissonRegressor(alpha=0.55, max_iter=800))])
    tree = lambda estimator: Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    return {
        "Poisson Regression": linear,
        "RF Goal Model": tree(RandomForestRegressor(n_estimators=160, max_depth=8, min_samples_leaf=5, random_state=random_state, n_jobs=-1)),
        "Extra Trees Goal Model": tree(ExtraTreesRegressor(n_estimators=180, max_depth=9, min_samples_leaf=4, random_state=random_state, n_jobs=-1)),
        "Gradient Goal Model": tree(GradientBoostingRegressor(n_estimators=110, learning_rate=0.04, max_depth=2, loss="huber", random_state=random_state)),
        "Hist Goal Model": tree(HistGradientBoostingRegressor(max_iter=150, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=1.0, random_state=random_state)),
    }


def _time_split(df, validation_fraction=0.22):
    df = df.sort_values("Date").reset_index(drop=True)
    cut = max(25, int(len(df) * (0.70 if len(df) < 80 else 1.0 - validation_fraction)))
    cut = min(cut, len(df) - 15)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _elo_probs_from_row(row):
    home_expect = float(np.clip(row.get("elo_home_expectancy", 0.5), 0.05, 0.95))
    draw_base = float(np.clip(row.get("league_draw_rate", 0.26), 0.16, 0.34))
    draw = float(np.clip(draw_base * np.exp(-abs(float(row.get("elo_diff", 0.0))) / 900.0), 0.12, 0.34))
    return np.array([(1.0 - draw) * home_expect, draw, (1.0 - draw) * (1.0 - home_expect)])


def _validation_elo_probs(df):
    return np.vstack([_elo_probs_from_row(r) for _, r in df.iterrows()])


def _validation_poisson_probs(df, dc=False):
    return np.vstack([matrix_1x2(score_matrix(r["lambda_form_home"], r["lambda_form_away"], dixon_coles_rho=-0.10 if dc else None)) for _, r in df.iterrows()])


def _binary_probability(model, X):
    probs, classes = model.predict_proba(X)[0], list(model.classes_)
    return float(probs[classes.index(1)]) if 1 in classes else 0.5


def _binary_validation_loss(model, X, y):
    probs, classes = model.predict_proba(X), list(model.classes_)
    p1 = probs[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    try:
        return float(log_loss(y, np.column_stack([1-p1, p1]), labels=[0, 1]))
    except Exception:
        return 9.0


@dataclass
class PredictionBundle:
    home_team: str
    away_team: str
    probabilities: Dict[str, float]
    expected_goals: Tuple[float, float]
    score_matrix: np.ndarray
    top_scores: List[Tuple[str, float]]
    method_table: pd.DataFrame
    validation_table: pd.DataFrame
    feature_snapshot: pd.DataFrame
    corners_probability: Optional[float]
    data_notes: List[str]


def train_and_predict(training_df: pd.DataFrame, prediction_row: pd.DataFrame, random_state=42) -> PredictionBundle:
    if training_df.empty or len(training_df) < 55:
        raise ValueError("At least about 55 causally engineered historical matches are needed for this model set.")
    training_df = training_df.sort_values("Date").reset_index(drop=True)
    pred = prediction_row.iloc[[0]].copy()
    home, away = str(pred.iloc[0]["HomeTeam"]), str(pred.iloc[0]["AwayTeam"])
    features = [c for c in FEATURE_COLUMNS if c in training_df.columns and c in pred.columns]
    X_all, X_pred = training_df[features], pred[features]
    train, val = _time_split(training_df)
    X_train, y_train, X_val, y_val = train[features], train["result"], val[features], val["result"]
    method_predictions, method_weights, validation_rows = {}, {}, []

    for name, dc in [("Form Poisson", False), ("Dixon-Coles", True)]:
        vp = _validation_poisson_probs(val, dc)
        loss = _safe_logloss(y_val, vp)
        acc = float(accuracy_score(y_val, np.array(RESULT_ORDER)[np.argmax(vp, axis=1)]))
        m = score_matrix(pred.iloc[0]["lambda_form_home"], pred.iloc[0]["lambda_form_away"], dixon_coles_rho=-0.10 if dc else None)
        method_predictions[name], method_weights[name] = matrix_1x2(m), _weight_from_logloss(loss)
        validation_rows.append({"Model": name, "Validation accuracy": acc, "Log loss": loss, "Type": "score distribution"})

    vp = _validation_elo_probs(val)
    loss = _safe_logloss(y_val, vp)
    acc = float(accuracy_score(y_val, np.array(RESULT_ORDER)[np.argmax(vp, axis=1)]))
    method_predictions["Elo"], method_weights["Elo"] = _elo_probs_from_row(pred.iloc[0]), _weight_from_logloss(loss)
    validation_rows.append({"Model": "Elo", "Validation accuracy": acc, "Log loss": loss, "Type": "rating"})

    for name, base in _base_classifier_models(random_state).items():
        try:
            model = clone(base).fit(X_train, y_train)
            vp = _class_probs_many(model, X_val)
            loss = _safe_logloss(y_val, vp)
            acc = float(accuracy_score(y_val, np.array(RESULT_ORDER)[np.argmax(vp, axis=1)]))
            final = clone(base).fit(X_all, training_df["result"])
            method_predictions[name], method_weights[name] = _class_probs(final, X_pred), _weight_from_logloss(loss)
            validation_rows.append({"Model": name, "Validation accuracy": acc, "Log loss": loss, "Type": "1X2 classifier"})
        except Exception as exc:
            validation_rows.append({"Model": name, "Validation accuracy": np.nan, "Log loss": np.nan, "Type": f"skipped: {exc}"})

    goal_matrices, lambda_methods = {}, {}
    for name, base in _goal_regressors(random_state).items():
        try:
            mh, ma = clone(base).fit(X_train, train["FTHG"]), clone(base).fit(X_train, train["FTAG"])
            vh, va = np.clip(mh.predict(X_val), 0.08, 5.5), np.clip(ma.predict(X_val), 0.08, 5.5)
            vp = np.vstack([matrix_1x2(score_matrix(h, a)) for h, a in zip(vh, va)])
            loss = _safe_logloss(y_val, vp)
            acc = float(accuracy_score(y_val, np.array(RESULT_ORDER)[np.argmax(vp, axis=1)]))
            fmh, fma = clone(base).fit(X_all, training_df["FTHG"]), clone(base).fit(X_all, training_df["FTAG"])
            lh, la = float(np.clip(fmh.predict(X_pred)[0], 0.08, 5.5)), float(np.clip(fma.predict(X_pred)[0], 0.08, 5.5))
            m = score_matrix(lh, la)
            goal_matrices[name], lambda_methods[name] = m, (lh, la)
            method_predictions[name], method_weights[name] = matrix_1x2(m), _weight_from_logloss(loss)
            validation_rows.append({"Model": name, "Validation accuracy": acc, "Log loss": loss, "Type": "goal regression"})
        except Exception as exc:
            validation_rows.append({"Model": name, "Validation accuracy": np.nan, "Log loss": np.nan, "Type": f"skipped: {exc}"})

    if len(method_predictions) < 5:
        raise RuntimeError("Too few prediction models trained successfully.")
    names = list(method_predictions)
    w = np.array([method_weights[n] for n in names]); w /= w.sum()
    ensemble_1x2 = np.sum(np.vstack([method_predictions[n] for n in names]) * w[:, None], axis=0)
    ensemble_1x2 /= ensemble_1x2.sum()

    score_matrices = [score_matrix(pred.iloc[0]["lambda_form_home"], pred.iloc[0]["lambda_form_away"]),
                      score_matrix(pred.iloc[0]["lambda_form_home"], pred.iloc[0]["lambda_form_away"], dixon_coles_rho=-0.10)]
    score_weights = [method_weights.get("Form Poisson", .2), method_weights.get("Dixon-Coles", .2)]
    for n, m in goal_matrices.items():
        score_matrices.append(m); score_weights.append(method_weights.get(n, .2))
    sw = np.array(score_weights); sw /= sw.sum()
    ensemble_matrix = np.sum(np.stack(score_matrices) * sw[:, None, None], axis=0); ensemble_matrix /= ensemble_matrix.sum()
    score_markets = matrix_markets(ensemble_matrix)

    binary_outputs = {}
    for target in ["btts", "over25"]:
        preds, weights = [], []
        for _, base in _binary_models(random_state).items():
            try:
                model = clone(base).fit(X_train, train[target].astype(int))
                loss = _binary_validation_loss(model, X_val, val[target].astype(int))
                final = clone(base).fit(X_all, training_df[target].astype(int))
                preds.append(_binary_probability(final, X_pred)); weights.append(_weight_from_logloss(loss))
            except Exception:
                pass
        if preds:
            bw = np.array(weights); bw /= bw.sum(); binary_outputs[target] = float(np.dot(preds, bw))
        else:
            binary_outputs[target] = np.nan
    btts_ml, o25_ml = binary_outputs.get("btts", np.nan), binary_outputs.get("over25", np.nan)
    btts = score_markets["BTTS"] if pd.isna(btts_ml) else .55 * score_markets["BTTS"] + .45 * btts_ml
    o25 = score_markets["O25"] if pd.isna(o25_ml) else .55 * score_markets["O25"] + .45 * o25_ml

    corners_prob = None
    corner_train = training_df[training_df["corners_over85"].notna()].copy()
    if len(corner_train) >= 80 and corner_train["corners_over85"].nunique() == 2:
        ct, cv = _time_split(corner_train); cp, cw = [], []
        for _, base in _binary_models(random_state).items():
            try:
                model = clone(base).fit(ct[features], ct["corners_over85"].astype(int))
                loss = _binary_validation_loss(model, cv[features], cv["corners_over85"].astype(int))
                final = clone(base).fit(corner_train[features], corner_train["corners_over85"].astype(int))
                cp.append(_binary_probability(final, X_pred)); cw.append(_weight_from_logloss(loss))
            except Exception:
                pass
        if cp:
            cw = np.array(cw); cw /= cw.sum(); corners_prob = float(np.dot(cp, cw))

    lambda_pairs = [(float(pred.iloc[0]["lambda_form_home"]), float(pred.iloc[0]["lambda_form_away"]))]
    lambda_w = [method_weights.get("Form Poisson", .2) + method_weights.get("Dixon-Coles", .2)]
    for n, pair in lambda_methods.items():
        lambda_pairs.append(pair); lambda_w.append(method_weights.get(n, .2))
    lw = np.array(lambda_w); lw /= lw.sum()
    exp_home = float(sum(weight * pair[0] for weight, pair in zip(lw, lambda_pairs)))
    exp_away = float(sum(weight * pair[1] for weight, pair in zip(lw, lambda_pairs)))

    top = sorted([(f"{h}-{a}", float(ensemble_matrix[h, a])) for h in range(ensemble_matrix.shape[0]) for a in range(ensemble_matrix.shape[1])], key=lambda x: x[1], reverse=True)
    total_weight = sum(method_weights.values())
    method_table = pd.DataFrame([{ "Method": n, "Home %": 100*method_predictions[n][0], "Draw %": 100*method_predictions[n][1], "Away %": 100*method_predictions[n][2], "Ensemble weight": 100*method_weights[n]/total_weight } for n in names]).sort_values("Ensemble weight", ascending=False).reset_index(drop=True)
    probs = {"home": float(ensemble_1x2[0]), "draw": float(ensemble_1x2[1]), "away": float(ensemble_1x2[2]), "btts_yes": float(np.clip(btts, 0, 1)), "over25": float(np.clip(o25, 0, 1))}
    notes = [f"Training rows: {len(training_df):,}; validation rows: {len(val):,}.", f"Successful 1X2 methods: {len(method_predictions)}.", "Validation uses a chronological holdout, not a random split.", "Bookmaker odds are not used as predictive features."]
    if corners_prob is None: notes.append("Corners O/U 8.5 unavailable because too few historical rows contain corner statistics.")
    return PredictionBundle(home, away, probs, (exp_home, exp_away), ensemble_matrix, top[:10], method_table,
        pd.DataFrame(validation_rows).sort_values("Log loss", na_position="last").reset_index(drop=True),
        pred[["Date", "league_code", "HomeTeam", "AwayTeam"] + features].T.reset_index().rename(columns={"index": "Feature", 0: "Value"}),
        corners_prob, notes)
