"""Stage 2 (situational adjustments), market blend, and outcome probabilities."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C

BASE_MARGIN = ["r_margin", "div_x_margin", "home_bye", "away_bye", "rest_diff",
               "away_west_early", "long_trip", "home_qb_change", "away_qb_change"]
BASE_TOTAL = ["r_total", "wind_excess", "cold", "dome", "qb_changes"]
EPA_MARGIN = ["epa_margin", "sr_margin"]
EPA_TOTAL = ["epa_sum", "sr_sum"]
PRECIP_TOTAL = -2.0   # no rain/snow history in the data, so this one is fixed


def margin_features() -> list[str]:
    return BASE_MARGIN + (EPA_MARGIN if C.USE_EPA else [])


def total_features() -> list[str]:
    return BASE_TOTAL + (EPA_TOTAL if C.USE_EPA else [])


def _design(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["div_x_margin"] = d["div_game"] * d["r_margin"]
    d["qb_changes"] = d["home_qb_change"] + d["away_qb_change"]
    # EPA in points per 100 plays, success rate in percentage points, so the
    # regression's coefficients sit on the same scale as the points features
    for src, dst in (("e_margin", "epa_margin"), ("e_sum", "epa_sum"), ("s_margin", "sr_margin"), ("s_sum", "sr_sum")):
        d[dst] = d[src].fillna(0.0) * 100 if src in d else 0.0
    return d


def _ols(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    pen = np.full(X1.shape[1], ridge)
    pen[0] = 1e-6
    return np.linalg.solve(X1.T @ X1 + np.diag(pen), X1.T @ y)


@dataclass
class SeasonModel:
    season: int
    b_margin: np.ndarray
    b_total: np.ndarray
    w_margin: float
    w_total: float
    sd_margin: float
    sd_total: float
    key_mult: np.ndarray = field(repr=False)
    ml_coef: tuple = (0.0, 0.14)
    n_train: int = 0

    # ---------- point predictions ----------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        d = _design(df)
        out = df.copy()
        Xm = np.column_stack([np.ones(len(d)), d[margin_features()].to_numpy(float)])
        Xt = np.column_stack([np.ones(len(d)), d[total_features()].to_numpy(float)])
        model_margin = Xm @ self.b_margin + (d.home_adj - d.away_adj).to_numpy()
        model_total = Xt @ self.b_total + 0.5 * (d.home_adj + d.away_adj).to_numpy() + PRECIP_TOTAL * d.precip.to_numpy()
        out["model_margin"] = model_margin
        out["model_total"] = model_total
        mm = d.mkt_margin.fillna(pd.Series(model_margin, index=d.index)).to_numpy()
        mt = d.mkt_total.fillna(pd.Series(model_total, index=d.index)).to_numpy()
        out["proj_margin"] = self.w_margin * model_margin + (1 - self.w_margin) * mm
        out["proj_total"] = self.w_total * model_total + (1 - self.w_total) * mt
        out["proj_home_pts"] = (out.proj_total + out.proj_margin) / 2
        out["proj_away_pts"] = (out.proj_total - out.proj_margin) / 2
        return out

    # ---------- distributions ----------
    def margin_pmf(self, mu: float) -> tuple[np.ndarray, np.ndarray]:
        k = np.arange(-80, 81)
        p = norm.pdf(k, mu, self.sd_margin) * self.key_mult[np.abs(k)]
        return k, p / p.sum()

    def spread_probs(self, mu: float, home_line: float) -> dict:
        """home_line: points the home team is favored by (nflverse spread_line)."""
        k, p = self.margin_pmf(mu)
        win = p[k > home_line].sum()
        push = p[np.isclose(k, home_line)].sum()
        return {"home": (win, push, 1 - win - push), "away": (1 - win - push, push, win)}

    def ml_probs(self, mu: float) -> dict:
        # Win probability from a logistic curve fit on history: a bell curve
        # underrates heavy favorites, so moneylines get their own calibration.
        a, b = self.ml_coef
        hw = 1 / (1 + np.exp(-(a + b * mu)))
        tie = 0.003
        hw = hw * (1 - tie)
        return {"home": (hw, tie, 1 - hw - tie), "away": (1 - hw - tie, tie, hw)}

    def total_probs(self, mu: float, line: float) -> dict:
        sd = self.sd_total
        if float(line).is_integer():
            over = 1 - norm.cdf(line + 0.5, mu, sd)
            under = norm.cdf(line - 0.5, mu, sd)
        else:
            over = 1 - norm.cdf(line, mu, sd)
            under = 1 - over
        push = 1 - over - under
        return {"over": (over, push, under), "under": (under, push, over)}


def key_number_multipliers(games: pd.DataFrame) -> np.ndarray:
    """How much more often each final margin happens than a smooth curve predicts (3, 7, 10...)."""
    res = games.loc[games.played, "result"].abs().round().astype(int)
    counts = np.bincount(res.clip(upper=80), minlength=81).astype(float)
    sym = np.concatenate([counts[:0:-1], counts])          # mirror to -80..80
    x = np.arange(-80, 81)
    kern = norm.pdf(np.arange(-12, 13), 0, C.KEY_NUMBER_SMOOTHING)
    smooth = np.convolve(sym, kern / kern.sum(), mode="same")
    mult_full = np.where(smooth > 0, (sym + 1) / (smooth + 1), 1.0)
    mult = mult_full[x >= 0]
    return np.clip(mult, 0.3, 3.0)


def fit_season_model(wf: pd.DataFrame, season: int, key_mult: np.ndarray) -> SeasonModel:
    """Train stage 2 and the market blend on all seasons before `season` (walk-forward)."""
    tr = wf[(wf.played) & (wf.season < season) & (wf.season >= C.STAGE2_MIN_SEASON)
            & wf.r_margin.notna() & wf.mkt_margin.notna() & wf.mkt_total.notna()]
    tr = _design(tr)
    b_m = _ols(tr[margin_features()].to_numpy(float), tr.result.to_numpy(float))
    b_t = _ols(tr[total_features()].to_numpy(float), tr.total.to_numpy(float))
    Xm = np.column_stack([np.ones(len(tr)), tr[margin_features()].to_numpy(float)])
    Xt = np.column_stack([np.ones(len(tr)), tr[total_features()].to_numpy(float)])
    mm, mt = Xm @ b_m, Xt @ b_t

    # Optimal blend weight: regress (actual - market) on (model - market), no intercept
    def blend(model, mkt, actual):
        dx, dy = model - mkt, actual - mkt
        return float(np.clip((dx @ dy) / (dx @ dx), 0, 1))

    w_m = blend(mm, tr.mkt_margin.to_numpy(), tr.result.to_numpy())
    w_t = blend(mt, tr.mkt_total.to_numpy(), tr.total.to_numpy())
    pm = w_m * mm + (1 - w_m) * tr.mkt_margin.to_numpy()
    pt = w_t * mt + (1 - w_t) * tr.mkt_total.to_numpy()
    return SeasonModel(season, b_m, b_t, w_m, w_t,
                       float(np.std(tr.result - pm)), float(np.std(tr.total - pt)),
                       key_mult, _logit_fit(pm, tr.result.to_numpy()), len(tr))


def _logit_fit(x: np.ndarray, result: np.ndarray) -> tuple[float, float]:
    keep = result != 0
    x, y = x[keep], (result[keep] > 0).astype(float)
    beta = np.zeros(2)
    X = np.column_stack([np.ones(len(x)), x])
    for _ in range(25):   # Newton-Raphson
        p = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (p * (1 - p))[:, None])
        beta += np.linalg.solve(H, X.T @ (y - p))
    return float(beta[0]), float(beta[1])
