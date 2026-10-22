"""Odds math, expected value, Kelly staking, and turning projections into bet options."""
from __future__ import annotations

import math

import pandas as pd

from . import config as C


def decimal(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def implied(american: float) -> float:
    return 1 / decimal(american)


def fair_american(p: float) -> float:
    if p <= 0 or p >= 1:
        return math.nan
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def ev(probs: tuple[float, float, float], american: float) -> float:
    """Expected profit per $1 staked. probs = (win, push, lose)."""
    w, _, l = probs
    return w * (decimal(american) - 1) - l


def kelly_stake(probs, american, bankroll=C.BANKROLL) -> float:
    w, _, l = probs
    b = decimal(american) - 1
    f = max(0.0, (b * w - l) / b)
    return bankroll * min(f * C.KELLY_FRACTION, C.MAX_STAKE_PCT)


def fmt_line(x: float) -> str:
    return "PK" if x == 0 else f"{x:+.1f}"


def game_options(g: pd.Series, sm) -> list[dict]:
    """All six sides for one game (spread, total, moneyline), with model probability and EV."""
    opts = []
    if pd.notna(g.spread_line):
        sp = sm.spread_probs(g.proj_margin, g.spread_line)
        opts += [
            dict(market="spread", side="home", pick=f"{g.home_team} {fmt_line(-g.spread_line)}",
                 probs=sp["home"], odds=g.home_spread_odds),
            dict(market="spread", side="away", pick=f"{g.away_team} {fmt_line(g.spread_line)}",
                 probs=sp["away"], odds=g.away_spread_odds),
        ]
    if pd.notna(g.total_line):
        tp = sm.total_probs(g.proj_total, g.total_line)
        opts += [
            dict(market="total", side="over", pick=f"Over {g.total_line}", probs=tp["over"], odds=g.over_odds),
            dict(market="total", side="under", pick=f"Under {g.total_line}", probs=tp["under"], odds=g.under_odds),
        ]
    if pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
        mp = sm.ml_probs(g.proj_margin)
        opts += [
            dict(market="moneyline", side="home", pick=f"{g.home_team} ML {int(g.home_moneyline):+d}",
                 probs=mp["home"], odds=g.home_moneyline),
            dict(market="moneyline", side="away", pick=f"{g.away_team} ML {int(g.away_moneyline):+d}",
                 probs=mp["away"], odds=g.away_moneyline),
        ]
    for o in opts:
        o["win_prob"] = o["probs"][0]
        o["ev"] = ev(o["probs"], o["odds"])
        o["stake"] = kelly_stake(o["probs"], o["odds"])
    return opts


def grade(opt: dict, g: pd.Series) -> float | None:
    """Profit per $1 for a settled game, None if not played."""
    if pd.isna(g.result):
        return None
    m, s = opt["market"], opt["side"]
    if m == "spread":
        diff = g.result - g.spread_line if s == "home" else g.spread_line - g.result
    elif m == "total":
        diff = g.total - g.total_line if s == "over" else g.total_line - g.total
    else:
        diff = g.result if s == "home" else -g.result
    if diff == 0:
        return 0.0
    return decimal(opt["odds"]) - 1 if diff > 0 else -1.0
