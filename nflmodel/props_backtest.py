"""Prop model track record: replay past weeks using only data available before each game.

For each week, team points come from the closing spread and total (so this measures the player model,
not the game model). Every projection is compared with what the player actually did.

  python run.py props-backtest                 # 2025 wk 5-18 and 2026 so far, current settings
  python run.py props-backtest --ablation      # also turn each adjustment off in turn to measure it
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import player_proj as PP

STAT_COL = {k: v[0] for k, v in PP.STATS.items()}


def _week_proj(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    g = games[(games.season == season) & (games.week == week) & games.result.notna() & games.spread_line.notna()].copy()
    g["proj_home_pts"] = (g.total_line + g.spread_line) / 2
    g["proj_away_pts"] = (g.total_line - g.spread_line) / 2
    return g


def _actuals(season: int) -> pd.DataFrame:
    s = PP._load("stats", season, False)
    s = s[s.season_type == "REG"].copy()
    for st, col in STAT_COL.items():
        s[st] = s[list(col)].fillna(0).sum(axis=1) if isinstance(col, tuple) else s[col].fillna(0)
    s["anytime_td"] = (s["anytime_td"] > 0).astype(float)
    return s[["season", "week", "player_display_name", "team"] + list(STAT_COL)]


def run(games: pd.DataFrame, weeks: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    acts = {}
    for season, week in weeks:
        proj = _week_proj(games, season, week)
        if proj.empty:
            continue
        p = PP.project(season, week, proj, refresh=False, games=games)
        if p.empty:
            continue
        if season not in acts:
            acts[season] = _actuals(season)
        a = acts[season]
        a = a[a.week == week].melt(id_vars=["season", "week", "player_display_name", "team"], var_name="stat", value_name="actual")
        m = p.merge(a, left_on=["player", "team", "stat"], right_on=["player_display_name", "team", "stat"], how="inner")
        m["season"], m["week"] = season, week
        rows.append(m[["season", "week", "player", "position", "stat", "projection", "actual"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score(r: pd.DataFrame) -> pd.DataFrame:
    out = []
    for st, g in r.groupby("stat"):
        if st == "anytime_td":
            p = g.projection.clip(0.01, 0.99)
            out.append(dict(stat=st, n=len(g), mae=np.nan, bias=float((g.actual - p).mean()),
                            brier=float(((p - g.actual) ** 2).mean()),
                            logloss=float(-(g.actual * np.log(p) + (1 - g.actual) * np.log(1 - p)).mean()), cv=np.nan))
        else:
            err = g.actual - g.projection
            cv = float(err.std() / g.projection.mean())
            out.append(dict(stat=st, n=len(g), mae=float(err.abs().mean()), bias=float(err.mean()),
                            brier=np.nan, logloss=np.nan, cv=round(cv, 3)))
    return pd.DataFrame(out)


def baseline(r: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Avg miss of a plain average of each player's last 4 games, for the same player-games."""
    seasons = sorted(r.season.unique())
    allst = pd.concat([_actuals(s) for s in sorted(set(seasons) | {min(seasons) - 1})], ignore_index=True)
    allst = allst.sort_values(["season", "week"])
    out = []
    for st, g in r[r.stat != "anytime_td"].groupby("stat"):
        a = allst[["season", "week", "player_display_name", st]].rename(columns={st: "v"})
        a["prev4"] = a.groupby("player_display_name").v.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        m = g.merge(a, left_on=["season", "week", "player"], right_on=["season", "week", "player_display_name"], how="left")
        out.append(dict(stat=st, mae_base=float((m.actual - m.prev4).abs().mean())))
    td = r[r.stat == "anytime_td"]
    if len(td):
        a = allst[["season", "week", "player_display_name", "anytime_td"]].rename(columns={"anytime_td": "v"})
        a["prev"] = a.groupby("player_display_name").v.transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
        m = td.merge(a, left_on=["season", "week", "player"], right_on=["season", "week", "player_display_name"], how="left")
        p = m.prev.fillna(m.prev.mean()).clip(0.02, 0.98)
        out.append(dict(stat="anytime_td", mae_base=np.nan, brier_base=float(((p - m.actual) ** 2).mean())))
    return pd.DataFrame(out)


def default_weeks(games: pd.DataFrame, current: tuple[int, int]) -> list[tuple[int, int]]:
    s, w = current
    weeks = [(s - 1, k) for k in range(5, 19)]
    weeks += [(s, k) for k in range(3, w)]
    return weeks
