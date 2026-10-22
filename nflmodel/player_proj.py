"""Player prop projections from nflverse weekly stats.

For every team playing this week: the starting QB (from the game board, so QB overrides apply)
and the active RBs, WRs and TEs who get real usage. Each stat is a recency-weighted average of
the player's recent games, shrunk a little toward a typical player in the same role, then
scaled by game script (the team's projected points vs. its season average) in props.py.

Players on IR or otherwise inactive on this week's roster are left out; Out and Doubtful
players on the injury report are left out; Questionable players are kept and flagged.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
URLS = {
    "stats": REL + "/stats_player/stats_player_week_{season}.csv",
    "roster": REL + "/weekly_rosters/roster_weekly_{season}.csv",
    "injuries": REL + "/injuries/injuries_{season}.csv",
}

# stat -> (column(s) in the weekly stats file, distribution)
STATS = {
    "pass_yds": ("passing_yards", "normal"),
    "pass_tds": ("passing_tds", "poisson"),
    "rush_yds": ("rushing_yards", "normal"),
    "rec_yds": ("receiving_yards", "normal"),
    "receptions": ("receptions", "poisson"),
    "anytime_td": (("rushing_tds", "receiving_tds"), "td"),
}
BY_POS = {
    "QB": ["pass_yds", "pass_tds", "rush_yds"],
    "RB": ["rush_yds", "rec_yds", "receptions", "anytime_td"],
    "WR": ["rec_yds", "receptions", "anytime_td"],
    "TE": ["rec_yds", "receptions", "anytime_td"],
}
MIN_MEAN = {"pass_yds": 100, "pass_tds": 0.5, "rush_yds": 12, "rec_yds": 15, "receptions": 1.5, "anytime_td": 0.08}
RECENCY = 0.85          # each older game counts 15% less
PRIOR_SEASON_W = 0.5    # last season's games count half on top of recency
SHRINK_GAMES = 1.0      # pseudo-games of a typical player in the same role


def _load(kind: str, season: int, refresh: bool) -> pd.DataFrame | None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{kind}_{season}.csv"
    if refresh or not path.exists():
        try:
            df = pd.read_csv(URLS[kind].format(season=season), low_memory=False)
            df.to_csv(path, index=False)
            return df
        except Exception as e:
            print(f"  could not download {kind} {season} ({e})")
            if not path.exists():
                return None
    return pd.read_csv(path, low_memory=False)


def _value(rows: pd.DataFrame, stat: str) -> pd.Series:
    col = STATS[stat][0]
    if isinstance(col, tuple):
        return rows[list(col)].fillna(0).sum(axis=1)
    return rows[col].fillna(0)


def _weighted(rows: pd.DataFrame, stat: str, season: int) -> tuple[float, float]:
    """(weighted mean, total weight) over a player's games, newest first."""
    r = rows.sort_values(["season", "week"], ascending=False)
    w = RECENCY ** np.arange(len(r)) * np.where(r.season < season, PRIOR_SEASON_W, 1.0)
    v = _value(r, stat).to_numpy(float)
    return (float((w * v).sum() / w.sum()), float(w.sum())) if len(r) else (0.0, 0.0)


def project(season: int, week: int, proj: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """proj: this week's game projections (home/away team, QB names, projected points)."""
    cur = _load("stats", season, refresh)
    if cur is None:
        return pd.DataFrame()
    prev = _load("stats", season - 1, False)
    stats = pd.concat([d for d in (prev, cur) if d is not None], ignore_index=True)
    stats = stats[(stats.season_type == "REG") & ~((stats.season == season) & (stats.week >= week))]
    stats = stats[stats.position.isin(BY_POS)].copy()

    roster = _load("roster", season, refresh)
    inj = _load("injuries", season, refresh)
    active, injury = None, {}
    if roster is not None and len(roster):
        rw = roster[roster.week == min(week, roster.week.max())]
        active = set(rw.loc[rw.status == "ACT", "gsis_id"])
    if inj is not None and len(inj):
        iw = inj[(inj.season == season) & (inj.week == week)]
        for r in iw.itertuples():
            status = r.report_status if isinstance(r.report_status, str) else None
            practice = r.practice_status if isinstance(r.practice_status, str) else None
            injury[r.gsis_id] = (status, practice, r.report_primary_injury if isinstance(r.report_primary_injury, str)
                                 else (r.practice_primary_injury if isinstance(r.practice_primary_injury, str) else ""))

    # typical regular in each role, for shrinkage
    recent = stats[stats.season == season].copy()
    typical = {}
    for pos, sts in BY_POS.items():
        p = recent[recent.position == pos]
        for st in sts:
            per = p.assign(v=_value(p, st)).groupby("player_id").v.mean()
            per = per[per > 0]
            typical[(pos, st)] = float(per.quantile(0.6)) if len(per) else 0.0

    team_game = {}
    for g in proj.itertuples():
        team_game[g.home_team] = (g, "home")
        team_game[g.away_team] = (g, "away")

    out = []
    for team, (g, side) in team_game.items():
        qb_name = getattr(g, f"{side}_qb_name")
        opp = g.away_team if side == "home" else g.home_team
        game = f"{g.away_team} @ {g.home_team}"
        on_team = recent[(recent.team == team)].player_id.unique()
        for pid in on_team:
            rows = stats[stats.player_id == pid]
            last = rows.sort_values(["season", "week"]).iloc[-1]
            pos, name = last.position, last.player_display_name
            if last.team != team:
                continue                                   # traded away
            if active is not None and pid not in active:
                continue                                   # IR, inactive, released
            st_inj = injury.get(pid)
            if st_inj and st_inj[0] in ("Out", "Doubtful"):
                continue
            if pos == "QB" and name != qb_name:
                continue                                   # only this week's starter
            note = ""
            if st_inj:
                status, practice, what = st_inj
                if status:
                    note = f"{status} ({what})" if what else status
                elif practice and "Did Not" in practice:
                    note = f"Missed practice ({what})" if what else "Missed practice"
                elif practice and "Limited" in practice:
                    note = f"Limited in practice ({what})" if what else "Limited in practice"
            for st in BY_POS[pos]:
                mean, wsum = _weighted(rows, st, season)
                prior = typical.get((pos, st), mean)
                est = (mean * wsum + prior * SHRINK_GAMES) / (wsum + SHRINK_GAMES)
                if mean < MIN_MEAN[st] or est < MIN_MEAN[st]:
                    continue
                if st == "anytime_td":
                    est = 1 - math.exp(-est)               # P(at least one TD)
                out.append(dict(game=game, team=team, opp=opp, player=name, position=pos, stat=st,
                                projection=round(est, 3), injury=note, games=int(len(rows))))
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    # receivers: keep the top 3 WRs, 2 TEs and 2 RBs per team by usage so the list stays readable
    keep = []
    usage_stat = {"WR": "rec_yds", "TE": "rec_yds", "RB": "rush_yds"}
    limits = {"WR": 3, "TE": 2, "RB": 2}
    for (team, pos), grp in df.groupby(["team", "position"]):
        if pos == "QB":
            keep.append(grp)
            continue
        top = (grp[grp.stat == usage_stat[pos]].sort_values("projection", ascending=False)
               .player.head(limits[pos]))
        if pos == "RB":   # pass-catching backs with little rushing still count
            top = pd.concat([top, grp[grp.stat == "rec_yds"].sort_values("projection", ascending=False)
                             .player.head(1)]).drop_duplicates()
        keep.append(grp[grp.player.isin(top)])
    return pd.concat(keep, ignore_index=True)
