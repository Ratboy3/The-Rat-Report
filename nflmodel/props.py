"""Player prop pricer: your projection + game script from the team model -> P(over) and EV."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from . import config as C
from .betting import ev, kelly_stake


def price_props(props: pd.DataFrame, week_proj: pd.DataFrame, team_ppg: dict[str, float]) -> pd.DataFrame:
    """props columns: player, stat, team, projection, line, over_odds, under_odds
    stat: pass_yds | rush_yds | rec_yds | receptions | pass_tds | anytime_td
    For anytime_td, projection is your probability of scoring (0-1)."""
    team_pts = {}
    for g in week_proj.itertuples():
        team_pts[g.home_team] = g.proj_home_pts
        team_pts[g.away_team] = g.proj_away_pts

    rows = []
    for p in props.itertuples():
        factor = 1.0
        if p.team in team_pts and team_ppg.get(p.team):
            factor = 1 + C.PROP_SCRIPT_WEIGHT * (team_pts[p.team] / team_ppg[p.team] - 1)
        if p.stat == "anytime_td":
            lam = -math.log(1 - min(0.95, max(0.01, p.projection))) * factor
            adj = 1 - math.exp(-lam)
            over = (adj, 0.0, 1 - adj)
        elif p.stat in C.PROP_CV:
            adj = p.projection * factor
            sd = max(1.0, adj * C.PROP_CV[p.stat])
            o = 1 - norm.cdf(p.line, adj, sd)
            over = (o, 0.0, 1 - o)
        elif p.stat in C.PROP_COUNT:
            adj = max(0.01, p.projection * factor)
            k = math.floor(p.line)
            if float(p.line).is_integer():
                push = poisson.pmf(k, adj)
                over = (1 - poisson.cdf(k, adj), push, poisson.cdf(k - 1, adj))
            else:
                over = (1 - poisson.cdf(k, adj), 0.0, poisson.cdf(k, adj))
        else:
            raise ValueError(f"unknown stat '{p.stat}' for {p.player}")
        under = (over[2], over[1], over[0])
        e_o, e_u = ev(over, p.over_odds), ev(under, p.under_odds)
        best, probs, odds = ("Over", over, p.over_odds) if e_o >= e_u else ("Under", under, p.under_odds)
        rows.append(dict(player=p.player, stat=p.stat, team=p.team, projection=p.projection,
                         script_factor=round(factor, 3), adj_projection=round(adj, 2), line=p.line,
                         p_over=round(over[0], 3), best_side=best, ev=round(max(e_o, e_u), 4),
                         stake=round(kelly_stake(probs, odds), 2),
                         flag=max(e_o, e_u) >= C.MIN_EV))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ automatic props
def _script_factor(team: str, team_pts: dict, team_ppg: dict) -> float:
    if team in team_pts and team_ppg.get(team):
        return 1 + C.PROP_SCRIPT_WEIGHT * (team_pts[team] / team_ppg[team] - 1)
    return 1.0


def _dist(stat: str, adj: float) -> dict:
    """Distribution the page's line checker uses: normal (mean, sd), poisson (lam) or td (p)."""
    if stat == "anytime_td":
        return dict(dist="td", a=adj, b=0.0)
    if stat in C.PROP_CV:
        return dict(dist="normal", a=adj, b=max(1.0, adj * C.PROP_CV[stat]))
    return dict(dist="poisson", a=adj, b=0.0)


def _probs(d: dict, line: float) -> tuple[float, float, float]:
    """(over, push, under) at a line."""
    if d["dist"] == "td":
        return (d["a"], 0.0, 1 - d["a"])
    if d["dist"] == "normal":
        o = 1 - norm.cdf(line, d["a"], d["b"])
        return (o, 0.0, 1 - o)
    k = math.floor(line)
    if float(line).is_integer():
        return (1 - poisson.cdf(k, d["a"]), poisson.pmf(k, d["a"]), poisson.cdf(k - 1, d["a"]))
    return (1 - poisson.cdf(k, d["a"]), 0.0, poisson.cdf(k, d["a"]))


def _fair_line(d: dict) -> float | None:
    if d["dist"] == "td":
        return None
    if d["dist"] == "normal":
        return round(d["a"] * 2) / 2 if round(d["a"] * 2) % 2 == 1 else math.floor(d["a"]) + 0.5
    best = min((abs(_probs(d, k + 0.5)[0] - 0.5), k + 0.5) for k in range(0, 15))
    return best[1]


def price_auto(players: pd.DataFrame, week_proj: pd.DataFrame, team_ppg: dict,
               lines: pd.DataFrame | None) -> pd.DataFrame:
    """players: output of player_proj.project. lines: player, stat, line, over_odds, under_odds, source, as_of."""
    if players is None or players.empty:
        return pd.DataFrame()
    team_pts = {}
    for g in week_proj.itertuples():
        team_pts[g.home_team] = g.proj_home_pts
        team_pts[g.away_team] = g.proj_away_pts
    book = {}
    if lines is not None and len(lines):
        for r in lines.itertuples():
            book[(str(r.player).strip().lower(), str(r.stat).strip())] = r
    rows = []
    for p in players.itertuples():
        f = _script_factor(p.team, team_pts, team_ppg)
        if p.stat == "anytime_td":
            lam = -math.log(1 - min(0.95, max(0.01, p.projection))) * f
            adj = 1 - math.exp(-lam)
        else:
            adj = p.projection * f
        d = _dist(p.stat, adj)
        row = dict(game=p.game, team=p.team, opp=p.opp, player=p.player, position=p.position, stat=p.stat,
                   injury=p.injury, script_factor=round(f, 3), adj_projection=round(adj, 3),
                   fair_line=_fair_line(d), dist=d["dist"], a=round(d["a"], 4), b=round(d["b"], 4),
                   line=np.nan, over_odds=np.nan, under_odds=np.nan, source="", as_of="",
                   p_over=np.nan, best_side="", ev=np.nan, stake=0.0, flag=False)
        b = book.get((p.player.lower(), p.stat))
        if b is not None and (pd.notna(b.over_odds) or pd.notna(b.under_odds)):
            line = 0.5 if p.stat == "anytime_td" else float(b.line)
            over = _probs(d, line)
            under = (over[2], over[1], over[0])
            oo = b.over_odds if pd.notna(b.over_odds) else None
            uo = b.under_odds if pd.notna(b.under_odds) else None
            e_o = ev(over, oo) if oo is not None else -1.0
            e_u = ev(under, uo) if uo is not None else -1.0
            best, probs, odds = ("Over", over, oo) if e_o >= e_u else ("Under", under, uo)
            row.update(line=line, over_odds=oo if oo is not None else np.nan, under_odds=uo if uo is not None else np.nan,
                       source=str(b.source), as_of=str(b.as_of), p_over=round(over[0], 3), best_side=best,
                       ev=round(max(e_o, e_u), 4), stake=round(kelly_stake(probs, odds), 2),
                       flag=max(e_o, e_u) >= C.MIN_EV)
        rows.append(row)
    return pd.DataFrame(rows)
