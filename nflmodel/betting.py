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


def suggested_units(american) -> float:
    """Suggested size in units: 1u, or less at long plus-money prices so a win pays about UNIT_LONGSHOT_WIN units."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return 1.0
    if a != a or a <= C.UNIT_LONGSHOT_ODDS:
        return 1.0
    return max(0.1, round(C.UNIT_LONGSHOT_WIN / (decimal(a) - 1), 1))


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


def shop(g: pd.Series, opts: list[dict], sm, alt: dict, name: str = "FanDuel") -> list[dict]:
    """Re-price each game bet at a second book (FanDuel) and keep whichever book's line and price is worth more.
    DraftKings' line still drives the model; this only changes where a bet is placed."""
    key = f"{g.away_team} @ {g.home_team}"
    for o in opts:
        o.setdefault("book", "DraftKings")
        a = alt.get((key, o["market"], o["side"]))
        if not a or pd.isna(a[1]):
            continue
        line, odds = a
        m, s = o["market"], o["side"]
        if m == "spread":
            if pd.isna(line):
                continue
            sl = -float(line) if s == "home" else float(line)          # nflverse convention: + = home favored
            probs = sm.spread_probs(g.proj_margin, sl)[s]
            pick = f"{g.home_team} {fmt_line(float(line))}" if s == "home" else f"{g.away_team} {fmt_line(float(line))}"
        elif m == "total":
            if pd.isna(line):
                continue
            sl = float(line)
            probs = sm.total_probs(g.proj_total, sl)[s]
            pick = f"{'Over' if s == 'over' else 'Under'} {sl:g}"
        else:
            sl = None
            probs = sm.ml_probs(g.proj_margin)[s]
            pick = f"{g.home_team if s == 'home' else g.away_team} ML {int(odds):+d}"
        e = ev(probs, float(odds))
        if e > o["ev"] + 1e-9:
            o.update(probs=probs, odds=float(odds), win_prob=probs[0], ev=e, stake=kelly_stake(probs, float(odds)),
                     pick=f"{pick} ({name})", book=name, line=sl)
    return opts
