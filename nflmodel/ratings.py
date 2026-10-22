"""Stage 1: offense/defense ratings from points, fit by time-weighted ridge regression.

For every game:  home_pts = mu + hfa + off[home] - def[away]
                 away_pts = mu       + off[away] - def[home]
Recent games count more (exponential decay), each offseason cuts weight further,
and the ridge penalty pulls every team toward league average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Ratings:
    teams: list[str]
    mu: float
    hfa: float
    off: dict[str, float]
    deff: dict[str, float]

    def net(self, t: str) -> float:
        return self.off.get(t, 0.0) + self.deff.get(t, 0.0)

    def table(self) -> pd.DataFrame:
        rows = [{"team": t, "off": self.off[t], "def": self.deff[t], "net": self.net(t)} for t in self.teams]
        return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)

    def predict(self, home: str, away: str, neutral: int) -> tuple[float, float]:
        h = self.mu + (0 if neutral else self.hfa) + self.off.get(home, 0) - self.deff.get(away, 0)
        a = self.mu + self.off.get(away, 0) - self.deff.get(home, 0)
        return h, a


def fit_ratings(games: pd.DataFrame, cutoff: pd.Timestamp, season: int) -> Ratings:
    hist = games[(games.played) & (games.date < cutoff) & (games.season >= season - C.LOOKBACK_SEASONS)]
    teams = sorted(set(hist.home_team) | set(hist.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(hist), len(teams)
    if n < 20:
        return Ratings(teams, 22.0, 1.5, {t: 0.0 for t in teams}, {t: 0.0 for t in teams})

    age = (cutoff - hist.date).dt.days.to_numpy()
    w = 0.5 ** (age / C.HALF_LIFE_DAYS) * C.OFFSEASON_DECAY ** (season - hist.season.to_numpy())

    # Columns: mu, hfa, off[T], def[T]
    X = np.zeros((2 * n, 2 + 2 * T))
    y = np.empty(2 * n)
    hi = hist.home_team.map(idx).to_numpy()
    ai = hist.away_team.map(idx).to_numpy()
    r = np.arange(n)
    X[r, 0] = 1
    X[r, 1] = 1 - hist.neutral.to_numpy()
    X[r, 2 + hi] = 1
    X[r, 2 + T + ai] = -1
    X[n + r, 0] = 1
    X[n + r, 2 + ai] = 1
    X[n + r, 2 + T + hi] = -1
    y[:n] = hist.home_score.to_numpy()
    y[n:] = hist.away_score.to_numpy()
    W = np.concatenate([w, w])

    penalty = np.full(2 + 2 * T, C.RIDGE_LAMBDA)
    penalty[:2] = 1e-6
    A = X.T @ (X * W[:, None]) + np.diag(penalty)
    b = X.T @ (W * y)
    beta = np.linalg.solve(A, b)
    off = {t: beta[2 + i] for t, i in idx.items()}
    deff = {t: beta[2 + T + i] for t, i in idx.items()}
    # center so ratings read as "points vs average"
    mo, md = np.mean(list(off.values())), np.mean(list(deff.values()))
    off = {t: v - mo for t, v in off.items()}
    deff = {t: v - md for t, v in deff.items()}
    return Ratings(teams, beta[0] + mo - md, beta[1], off, deff)


@dataclass
class PlayRatings:
    """Per-play efficiency ratings (EPA/play or success rate), offense and defense."""
    mu: float
    hfa: float
    off: dict[str, float]
    deff: dict[str, float]

    def expected(self, team: str, opp: str, home: bool) -> float:
        """Expected value for `team`'s offense vs `opp`, relative to league average."""
        return (self.hfa if home else 0.0) + self.off.get(team, 0.0) - self.deff.get(opp, 0.0)


def attach_epa(epa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Add game date and neutral-site flag to the team-game EPA table."""
    info = games[["game_id", "date", "neutral"]]
    return epa.merge(info, on="game_id", how="inner")


def fit_play_ratings(tg: pd.DataFrame, metric: str, cutoff: pd.Timestamp, season: int) -> PlayRatings:
    hist = tg[(tg.date < cutoff) & (tg.season >= season - C.LOOKBACK_SEASONS)].dropna(subset=[metric])
    teams = sorted(set(hist.team) | set(hist.opp))
    if len(hist) < 40:
        return PlayRatings(0.0, 0.0, {}, {})
    idx = {t: i for i, t in enumerate(teams)}
    T, n = len(teams), len(hist)
    age = (cutoff - hist.date).dt.days.to_numpy()
    w = 0.5 ** (age / C.HALF_LIFE_DAYS) * C.OFFSEASON_DECAY ** (season - hist.season.to_numpy())
    X = np.zeros((n, 2 + 2 * T))
    r = np.arange(n)
    X[r, 0] = 1
    X[r, 1] = hist.is_home.to_numpy() * (1 - hist.neutral.to_numpy())
    X[r, 2 + hist.team.map(idx).to_numpy()] = 1
    X[r, 2 + T + hist.opp.map(idx).to_numpy()] = -1
    y = hist[metric].to_numpy()
    # scale the penalty to the metric's own noise so one lambda fits every metric
    scale = np.var(y) / 100.0
    pen = np.full(2 + 2 * T, C.EPA_RIDGE_LAMBDA * scale)
    pen[:2] = 1e-9
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(pen), X.T @ (w * y))
    off = {t: beta[2 + i] for t, i in idx.items()}
    deff = {t: beta[2 + T + i] for t, i in idx.items()}
    mo, md = np.mean(list(off.values())), np.mean(list(deff.values()))
    return PlayRatings(beta[0] + mo - md, beta[1], {t: v - mo for t, v in off.items()},
                       {t: v - md for t, v in deff.items()})


PLAY_METRICS = {"epa": "e", "sr": "s"}


def walk_forward(games: pd.DataFrame, start_season: int, epa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rating-only predictions for every game from start_season on, using only prior data.

    Adds points-based columns (r_*) and, when play-by-play is given, efficiency
    columns: e_margin / e_sum (EPA per play) and s_margin / s_sum (success rate).
    """
    out = games.copy()
    for c in ["r_home", "r_away"] + [f"{p}_{k}" for p in PLAY_METRICS.values() for k in ("margin", "sum")]:
        out[c] = np.nan
    tg = attach_epa(epa, games) if epa is not None else None
    target = out[out.season >= start_season]
    for (season, week), grp in target.groupby(["season", "week"], sort=True):
        cutoff = grp.date.min()
        rt = fit_ratings(games, cutoff, season)
        prs = {}
        if tg is not None:
            prs = {m: fit_play_ratings(tg, m, cutoff, season) for m in PLAY_METRICS}
        for i, g in grp.iterrows():
            h, a = rt.predict(g.home_team, g.away_team, g.neutral)
            out.at[i, "r_home"], out.at[i, "r_away"] = h, a
            home = not g.neutral
            for m, pre in PLAY_METRICS.items():
                if m not in prs or not prs[m].off:
                    continue
                eh = prs[m].expected(g.home_team, g.away_team, home)
                ea = prs[m].expected(g.away_team, g.home_team, False)
                out.at[i, f"{pre}_margin"], out.at[i, f"{pre}_sum"] = eh - ea, eh + ea
    out["r_margin"] = out.r_home - out.r_away
    out["r_total"] = out.r_home + out.r_away
    return out
