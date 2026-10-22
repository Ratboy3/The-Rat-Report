"""Player prop pricer: your projection + game script from the team model -> P(over) and EV."""
from __future__ import annotations

import math

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
