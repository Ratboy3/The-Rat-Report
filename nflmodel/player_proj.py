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
    "QB": ["pass_yds", "pass_tds", "rush_yds", "anytime_td"],
    "RB": ["rush_yds", "rec_yds", "receptions", "anytime_td"],
    "WR": ["rec_yds", "receptions", "anytime_td"],
    "TE": ["rec_yds", "receptions", "anytime_td"],
}
MIN_MEAN = {"pass_yds": 100, "pass_tds": 0.5, "rush_yds": 12, "rec_yds": 15, "receptions": 1.5, "anytime_td": 0.0}
RECENCY = 0.85          # each older game counts 15% less
PRIOR_SEASON_W = 0.5    # last season's games count half on top of recency
SHRINK_GAMES = 1.0      # pseudo-games of a typical player in the same role
SHRINK_TD_GAMES = 2.5   # touchdowns are streaky, so lean harder on the typical player


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


XTD_PER_CARRY = 0.030   # league-typical touchdowns per carry
XTD_PER_TARGET = 0.045  # and per target
TD_HISTORY_W = 0.35     # weight on a player's actual TD rate vs. the rate his usage implies


RZ_TD = {"rz_carries": 0.10, "gl_carries": 0.25, "rz_targets": 0.14, "gl_targets": 0.20}  # TDs per opportunity
RZ_WEIGHT = 0.3         # red-zone rate vs. overall usage: a wash in testing (Brier 0.1901 vs 0.1903), kept light


def _zero_games(rows: pd.DataFrame, pid: str, team: str, season: int, team_weeks: list, ros_act: set | None,
                out_inj: set | None) -> pd.DataFrame:
    """This season's team games where the player was on the active roster, not ruled out, and got no stats row:
    real zero-usage games (a backup who didn't play), so one spot start doesn't make him look like a starter."""
    if not team_weeks or ros_act is None:
        return rows
    played = set(rows[rows.season == season].week)
    add = [w for w in team_weeks if w not in played and (w, pid) in ros_act and (w, pid) not in (out_inj or set())]
    if not add:
        return rows
    z = pd.DataFrame({"season": season, "week": add, "team": team, "player_id": pid})
    return pd.concat([rows, z], ignore_index=True)


def _td_shares(stats: pd.DataFrame, season: int, team: str, ids: list, rz: pd.DataFrame | None = None,
               zero=None) -> dict:
    """Each player's expected share of his team's rushing + receiving touchdowns, from red-zone and overall
    usage plus actual scoring."""
    rate = {}
    for pid in ids:
        r = stats[stats.player_id == pid]
        if r.empty:
            continue
        if zero is not None:
            r = _zero_games(r, pid, team, season, *zero)
        r = r.sort_values(["season", "week"], ascending=False)
        w = RECENCY ** np.arange(len(r)) * np.where(r.season < season, PRIOR_SEASON_W, 1.0)
        if USE["newteam"]:   # usage logged for another team says less about his role here
            w = w * np.where(r.team.fillna(team).to_numpy() != team, NEWTEAM_W, 1.0)
        car = r.carries.fillna(0).to_numpy(float)
        tgt = r.targets.fillna(0).to_numpy(float)
        tds = (r.rushing_tds.fillna(0) + r.receiving_tds.fillna(0)).to_numpy(float)
        xtd = (w * (car * XTD_PER_CARRY + tgt * XTD_PER_TARGET)).sum() / w.sum()
        if rz is not None and len(rz):
            g = r[["season", "week"]].merge(rz[rz.player_id == pid], on=["season", "week"], how="left").fillna(0)
            rz_x = sum(g[c].to_numpy(float) * v for c, v in RZ_TD.items())
            rz_x = rz_x + (np.maximum(0, car - g.rz_carries.to_numpy(float)) * 0.006
                           + np.maximum(0, tgt - g.rz_targets.to_numpy(float)) * 0.02)   # long touchdowns
            xtd = (1 - RZ_WEIGHT) * xtd + RZ_WEIGHT * (w * rz_x).sum() / w.sum()
        act = (w * tds).sum() / w.sum()
        rate[pid] = max(0.0, (1 - TD_HISTORY_W) * xtd + TD_HISTORY_W * act)
    tot = sum(rate.values())
    return {k: v / tot for k, v in rate.items()} if tot > 0 else {}


def _td_per_point(stats: pd.DataFrame, games: pd.DataFrame | None, season: int, week: int = 99) -> float:
    """Rushing + receiving touchdowns per point scored, from recent seasons (about 0.10). Points and TDs come from
    the same games: only games before this week (the stats are already cut there)."""
    try:
        g = games[(games.season >= season - 1) & games.result.notna()
                  & ((games.season < season) | ((games.season == season) & (games.week < week)))
                  & (games.game_type == "REG")]
        pts = float(g.home_score.sum() + g.away_score.sum())
        s = stats[stats.season >= season - 1]
        tds = float(s.rushing_tds.fillna(0).sum() + s.receiving_tds.fillna(0).sum())
        return tds / pts if pts > 0 else 0.10
    except Exception:
        return 0.10


def project(season: int, week: int, proj: pd.DataFrame, refresh: bool = False, games: pd.DataFrame | None = None) -> pd.DataFrame:
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
            per = p.assign(v=_value(p, st)).groupby("player_id").v.agg(["mean", "size"])
            per = per[per["size"] >= 2]["mean"]
            if st == "anytime_td":   # average regular in the role, zeros included
                usage = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"}[pos]
                regs = p.groupby("player_id")[usage].mean()
                per = per[per.index.isin(regs[regs >= regs.quantile(0.5)].index)]
                typical[(pos, st)] = float(per.mean()) if len(per) else 0.0
            else:
                per = per[per > 0]
                typical[(pos, st)] = float(per.quantile(0.6)) if len(per) else 0.0

    tpp = _td_per_point(stats, games, season, week)
    team_pts = {}
    for g in proj.itertuples():
        team_pts[g.home_team] = g.proj_home_pts
        team_pts[g.away_team] = g.proj_away_pts

    from . import pbp_extra
    rz, tm = pbp_extra.load()
    tend = pbp_extra.tendencies(tm, season, week).set_index("team") if tm is not None else None
    role = _snap_roles(season, week, roster)
    opp_f = _opponent_factors(stats, season, week)
    opp_tab = _opponent_table(stats, season, week)    # raw allowed per game + rank, for "Why this bet"
    man, splits, lg_man = ({}, {}, 0.3)
    if USE["coverage"]:
        try:
            from . import coverage
            man, splits, lg_man = coverage.tables(season)
        except Exception as e:
            print(f"  coverage data skipped ({e})")
    press = _pressure(tm, season, week) if USE["pressure"] else {}

    routes_t = None
    if USE["routes"]:
        from . import routes as R
        routes_t = R.load()
        if routes_t is not None:
            col = "routes" if ROUTE_SRC == "exact" else "routes_proxy"
            rt = routes_t.copy()
            rt["r"] = rt[col].fillna(rt.routes_proxy) if col == "routes" else rt[col]
            routes_t = rt.dropna(subset=["r"])[["season", "week", "player_id", "r"]]
    coach_agg = {}
    if USE["agg"]:
        if "home_agg" not in proj and games is not None and "home_coach" in proj:
            from . import aggression
            proj = aggression.attach(proj, aggression.load_cached(games))
        if "home_agg" in proj:
            for g in proj.itertuples():
                coach_agg[g.home_team], coach_agg[g.away_team] = g.home_agg, g.away_agg

    team_game = {}
    for g in proj.itertuples():
        team_game[g.home_team] = (g, "home")
        team_game[g.away_team] = (g, "away")

    out = []
    for team, (g, side) in team_game.items():
        qb_name = getattr(g, f"{side}_qb_name")
        opp = g.away_team if side == "home" else g.home_team
        game = f"{g.away_team} @ {g.home_team}"
        on_team = list(recent[(recent.team == team)].player_id.unique())
        # players back from injury who haven't played this season yet (e.g. a starter coming off IR)
        if roster is not None and len(roster):
            rw_t = roster[(roster.week == min(week, roster.week.max())) & (roster.team == team) & (roster.status == "ACT")
                          & roster.position.isin(BY_POS)]
            prior = stats[(stats.season == season - 1) & stats.player_id.isin(set(rw_t.gsis_id))]
            back = prior.groupby("player_id").size()
            on_team += [pid for pid in back[back >= 4].index if pid not in set(on_team)]
        # players who can score this week: active, not Out/Doubtful, on this team, and (for QBs) the starter
        eligible = []
        for pid in on_team:
            last = stats[stats.player_id == pid].sort_values(["season", "week"]).iloc[-1]
            if last.team != team or (active is not None and pid not in active):
                continue
            inj_ = injury.get(pid)
            if inj_ and inj_[0] in ("Out", "Doubtful"):
                continue
            if last.position == "QB" and last.player_display_name != qb_name:
                continue
            eligible.append(pid)
        zero = None
        if USE["zerofill"] and roster is not None:
            ra = roster[(roster.team == team) & (roster.status == "ACT")]
            ros_act = set(zip(ra.week, ra.gsis_id))
            outs = set()
            if inj is not None and len(inj):
                io = inj[(inj.season == season) & (inj.team == team) & inj.report_status.isin(["Out", "Doubtful"])]
                outs = set(zip(io.week, io.gsis_id))
            tw = sorted(stats[(stats.season == season) & (stats.team == team)].week.unique())
            zero = (tw, ros_act, outs)
        shares = _td_shares(stats, season, team, eligible, rz, zero)
        redis = _redistribution(stats, season, week, team, on_team, set(eligible))
        vol = _volume_factors(tend, team, opp)
        team_tds = team_pts.get(team, 22.0) * tpp
        if USE["agg"]:
            team_tds = max(0.3, team_tds + AGG_TD_PER_10 * coach_agg.get(team, 0.0) / 10)
        team_weeks = sorted(stats[(stats.season == season) & (stats.team == team)].week.unique())
        qb_chg = bool(getattr(g, f"{side}_qb_change_t", getattr(g, f"{side}_qb_change", 0)) or 0)   # backup QB starting (a returning starter is not one)
        lead_out = _leads_out(recent, team, set(eligible)) if USE["nextman"] else set()
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
            # back after missing the team's last game (but a regular before that)
            pw = set(rows[rows.season == season].week)
            returning = bool(team_weeks) and team_weeks[-1] not in pw and len(pw & set(team_weeks[-5:-1])) >= 2
            for st in BY_POS[pos]:
                mean, wsum = _weighted(rows, st, season)
                why = {}                                   # factors behind the projection, for "Why this bet"
                prior = typical.get((pos, st), mean)
                k = SHRINK_TD_GAMES if st == "anytime_td" else SHRINK_GAMES
                est = (mean * wsum + prior * k) / (wsum + k)
                if mean < MIN_MEAN[st] or est < MIN_MEAN[st]:
                    continue
                if st != "anytime_td" and pos != "QB":
                    recent3 = float(_value(rows.sort_values(["season", "week"]).tail(3), st).mean())
                    if USE["recent3"]:
                        est = (1 - RECENT3_W) * est + RECENT3_W * recent3   # lean toward the last 3 games
                    if USE["role"]:
                        est *= role.get(pid, 1.0)                       # snap share trending up or down
                    if USE["redis"]:
                        est *= redis["rec" if st in ("rec_yds", "receptions") else "rush"]
                if USE["airyards"] and pos in ("WR", "TE", "RB") and st in ("rec_yds", "receptions"):
                    est = AIR_W * _usage_est(rows, st, season) + (1 - AIR_W) * est
                if USE["routes"] and pos in ("WR", "TE", "RB") and st in ("rec_yds", "receptions"):
                    est *= _route_factor(rows, routes_t, pid, pos, season)
                if USE["coverage"] and st == "rec_yds" and pos in ("WR", "TE"):
                    from .coverage import factor as cov_factor
                    est *= cov_factor(splits, man, lg_man, pid, opp, COVERAGE_W)
                if USE["pressure"] and pos == "QB" and st in ("pass_yds", "pass_tds"):
                    est *= press.get((team, opp), 1.0)
                if USE["returning"] and returning and st != "anytime_td":
                    est *= RETURN_FACTOR
                if st != "anytime_td":
                    why["base"] = est
                    why["avg"] = mean
                    why["l3"] = float(_value(rows.sort_values(["season", "week"]).tail(3), st).mean())
                    why.update(_why_detail(rows, st, pos, season, opp_tab.get((opp, pos, st))))
                    if USE["opp"]:
                        why["f_opp"] = opp_f.get((opp, pos, st), 1.0)
                        est *= why["f_opp"]
                    if USE["vol"]:
                        why["f_vol"] = vol["pass" if st in ("pass_yds", "pass_tds", "rec_yds", "receptions") else "rush"]
                        est *= why["f_vol"]
                    if USE["backupqb"] and qb_chg and pos != "QB":
                        why["f_bqb"] = BACKUP_QB.get(st, 1.0) if (st != "rush_yds" or pos == "RB") else 1.0
                        est *= why["f_bqb"]
                    if pos in lead_out:
                        why["f_next"] = NEXT_MAN.get((pos, st), 1.0)
                        est *= why["f_next"]
                    est *= SCALE.get(st, 1.0)
                    if st in SHRINK_TO:                 # pull extreme projections toward the league average
                        m0, k0 = SHRINK_TO[st]
                        est = m0 + k0 * (est - m0)
                if st == "anytime_td":
                    lam = shares.get(pid, 0.0) * team_tds  # this player's expected TDs this game
                    if lam < 0.06:
                        continue
                    est = 1 - math.exp(-lam)               # P(at least one TD)
                    why = {"share": shares.get(pid, 0.0), "team_tds": team_tds,
                           "td_games": int((rows.season == season).sum()),
                           "td_scored": int((_value(rows[rows.season == season], st) > 0).sum())}
                    why.update(_why_td_detail(rows, pid, season, rz))
                out.append(dict(game=game, team=team, opp=opp, player=name, position=pos, stat=st,
                                projection=round(est, 3), injury=note, games=int(len(rows)), returning=returning,
                                **{f"why_{k}": (v if isinstance(v, str) else round(float(v), 4)) for k, v in why.items()}))
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


# switches the prop backtest uses to measure each factor (props_backtest.py)
# tested on 2025-26 (props-backtest): opponent defense and pace/pass mix improved projections; recent-3-game
# weighting, snap-share trend and teammate-usage redistribution made them worse, so they are off.
# Second round (Oct 22): air yards + targets helped receiving yards in both seasons (miss 22.71 -> 22.48) and is on.
# Man/zone coverage matchups (last season's charting) were neutral, pass rush vs. protection made passing yards
# worse (57.9 -> 59.1; the opponent factor already covers it), and a first-game-back discount held in 2025
# (-18%) but reversed in 2026 (+5%), so those three are off.
# Third round (Oct 22): head-coach 4th-down aggressiveness. Teams with aggressive coaches score more TDs than
# their market-implied points suggest: +0.185 TDs per team-game per 10 pct pts of go rate over expected
# (2010-24 fit, t~3; 2018+ t~3.6). Two-point tendencies added nothing. But adding it to the TD props made
# them slightly worse in the 2025-26 replay (Brier 0.1950 -> 0.1955, both seasons worse): players' own TD
# history already carries their coach's style. Off; the rating is shown on game cards for information.
# Fourth round (Oct 23): targets per route run x expected routes. Off: receiving yards got worse (miss 22.57 ->
# 22.62 at half weight, 22.82 at full; worse for WRs and in 2026), receptions about even (1.646 -> 1.639, under 2
# standard errors, worse in 2026). Exact routes from last season's participation data did no better than the
# snap-share proxy, so the missing 2026 participation file isn't what's holding it back.
# Fifth round (Oct 23), touchdowns: fixed a bug where TDs per point counted points from games after the replayed
# week (and postseason points live), which pushed every TD chance down (replay avg 19.7% vs 27.5% actual). Fixed:
# Brier 0.1956 -> 0.1882 (both seasons better), avg 26.7% vs 27.5% actual. Live TDs per point 0.102 -> 0.106.
# Tested and off: counting active-but-unused games as zeros (neutral: better 2025, worse 2026) and discounting
# games for a previous team (slightly worse at 35% and 15% weight).
USE = {"recent3": False, "role": False, "redis": False, "opp": True, "vol": True,
       "airyards": True, "coverage": False, "pressure": False, "returning": False, "agg": False,
       "routes": False, "zerofill": False, "newteam": False, "backupqb": True, "nextman": True}
# Round 7 (Oct 27, 2026), replay 2025 wk5-18 + 2026 wk3-7, each factor fit on one season and checked on the other
# (log loss of the over/under at the current projection's median line, on the affected player-games):
# Backup QB starting (the model's QB-change flag): the team's receivers fall short of their receiving-yards
# projection (actual/projected 0.925 in 2025, 0.953 in 2026) and its running backs out-rush theirs (1.07 / 1.19).
# Out of sample: receiving yds 0.6920 -> 0.6848 (2025 fit, 2026 test) and 0.6923 -> 0.6890 (2026 fit, 2025 test);
# RB rushing 0.6934 -> 0.6928 and 0.6935 -> 0.6833. Receptions were mixed and QB passing yards got worse (no change).
BACKUP_QB = {"rec_yds": 0.93, "rush_yds": 1.10}
# Next man up: when a team's top-usage player at a position (carries + targets for RBs, targets for WR/TE; 2+ games
# this season) is out, his same-position teammates beat their projections: RB rushing yds x1.50 in both seasons,
# receptions x1.10 (1.10 / 1.14). Out of sample: RB rushing 0.6970 -> 0.6943 and 0.6931 -> 0.6432; receptions
# 0.7023 -> 0.6915 and 0.7094 -> 0.6966. Receiving yards were mixed (better one way, worse the other): no change.
# The broader version (every injured teammate's share redistributed, "redis") tested worse and stays off.
NEXT_MAN = {("RB", "rush_yds"): 1.50, ("RB", "receptions"): 1.10, ("WR", "receptions"): 1.10, ("TE", "receptions"): 1.10}
# Wind and cold (Round 7): not used. Windy games (12+ mph outdoors) did see lower passing and receiving in 2025
# (x0.90 / x0.87), but the game total already prices the weather and the factor made 2026 worse out of sample
# (passing yds 0.6955 -> 0.8254 on the 6 windy QB-games so far; receiving 0.6958 -> 0.7112). Retest in December.


def _leads_out(recent: pd.DataFrame, team: str, eligible: set) -> set:
    """Positions (RB/WR/TE) whose top-usage player this season (2+ games) is not available this week."""
    r = recent[(recent.team == team) & recent.position.isin(["RB", "WR", "TE"])]
    if r.empty:
        return set()
    use = np.where(r.position == "RB", r.carries.fillna(0) + r.targets.fillna(0), r.targets.fillna(0))
    u = r.assign(u=use).groupby(["position", "player_id"]).agg(u=("u", "mean"), n=("week", "nunique")).reset_index()
    u = u[u.n >= 2]
    out = set()
    for pos, x in u.groupby("position"):
        lead = x.sort_values("u").iloc[-1].player_id
        if lead not in eligible:
            out.add(pos)
    return out
# Calibration: QBs beat the passing-yards projection on average (2025 actual/projected = 1.031). Applied to 2026
# (out of sample) it cut the bias from +10.2 to +3.3 yards and the average miss from 57.4 to 57.1. The same
# fix for rushing, receiving and receptions did not help on 2026, so those stay at 1.0. (Oct 26, 2026)
SCALE = {"pass_yds": 1.031}
# Passing-yards projections were too spread out: QBs projected under 170 yards beat it by ~+30 on average (Week 7:
# Huntley projected 149, threw 298; McCarthy 160, threw 252) and QBs projected over 260 fell ~13 short, in both
# seasons. Pull each projection toward 228 yards, keeping 70% of its distance (est = 228 + 0.70 x (est - 228)).
# Walk-forward: slope fit on 2025 (0.59) cut the 2026 average miss 56.70 -> 56.24; fit on 2026 (0.70) cut 2025's
# 58.79 -> 57.68. Shipped 0.70/228: 2025 58.79 -> 57.73, 2026 56.70 -> 55.85; Week 7 real DraftKings lines
# log loss 0.7103 -> 0.7061 (DraftKings 0.6935), and the 4 losing passing-yards bets would not have been flagged.
# (Oct 27, 2026)
SHRINK_TO = {"pass_yds": (228.0, 0.70)}
NEWTEAM_W = 0.35         # weight on a player's games for a previous team, in touchdown shares
# The Tuesday props-backtest --agg-test writes agg_test.csv; its verdict sets the switch.
try:
    _t = pd.read_csv(Path(__file__).resolve().parent.parent / "agg_test.csv")
    USE["agg"] = bool(str(_t.use_agg.iloc[-1]).strip().lower() == "true")
except Exception:
    pass
ROUTE_SRC = "proxy"      # "exact" (participation, last season only) or "proxy" (snap share x team dropbacks)
ROUTE_W = 0.5            # how much of the route-based target change carries into the projection
ROUTE_RECENCY = 0.7      # routes reflect role, which changes faster than talent: steeper recency
TPRR_PRIOR = {"WR": 0.20, "TE": 0.17, "RB": 0.16}   # targets per route run of a typical player
TPRR_SHRINK = 80         # routes of prior
AGG_TD_PER_10 = 0.18    # extra team TDs per game per +10 pct pts of coach go-for-it rate over expected
AIR_W = 0.5             # weight on the usage-based estimate (air yards + targets) for receiving props
AIR_YDS_COEF, TGT_YDS_COEF = 0.204, 5.812   # receiving yards ~ 0.204 x air yards + 5.8 x targets (fit on 2025)
LG_CATCH, CATCH_SHRINK = 0.675, 20
COVERAGE_W = 0.5
PRESSURE_W = 0.3
RETURN_FACTOR = 1.0     # set from the backtest
RECENT3_W = 0.3

ROLE_WEIGHT = 0.6       # how much of a snap-share change carries into the projection
OPP_SHRINK = 0.5        # opponent effects are noisy: use half of what the data shows
REDIS_WEIGHT = 0.7      # share of an injured teammate's targets/carries that go to the remaining players


def _snap_roles(season: int, week: int, roster) -> dict:
    """gsis_id -> factor from the player's last 2 games' snap share vs. his usual share."""
    from . import injuries
    snaps = injuries._load_snaps(season, False)
    if snaps is None or roster is None:
        return {}
    pfr = dict(zip(roster.pfr_id, roster.gsis_id))
    sn = snaps[(snaps.game_type == "REG") & (snaps.week < week) & (snaps.offense_pct > 0)].sort_values("week")
    out = {}
    for pid, g in sn.groupby("pfr_player_id"):
        if pid not in pfr or len(g) < 3:
            continue
        usual = g.offense_pct.iloc[:-2].tail(6).mean()
        last2 = g.offense_pct.tail(2).mean()
        if usual > 0.05:
            out[pfr[pid]] = float(np.clip(1 + ROLE_WEIGHT * (last2 / usual - 1), 0.6, 1.3))
    return out


def _opponent_table(stats: pd.DataFrame, season: int, week: int) -> dict:
    """(opponent, position, stat) -> (allowed per game over its last 8 games, rank 1 = most allowed, teams ranked).
    Same window as _opponent_factors; only used to explain bets."""
    s = stats[(stats.season == season) | ((stats.season == season - 1) & (stats.week >= 12))]
    out = {}
    for stat in ("rush_yds", "rec_yds", "receptions", "pass_yds", "pass_tds"):
        col = STATS[stat][0]
        g = s.groupby(["opponent_team", "position", "season", "week"])[col].sum().reset_index()
        g = g.sort_values(["season", "week"]).groupby(["opponent_team", "position"]).tail(8)
        per = g.groupby(["opponent_team", "position"])[col].mean()
        for pos, grp in per.groupby(level="position"):
            rk = grp.rank(ascending=False, method="min")
            for (opp, _), v in grp.items():
                out[(opp, pos, stat)] = (float(v), int(rk[(opp, pos)]), int(len(grp)))
    return out


def _why_detail(rows: pd.DataFrame, st: str, pos: str, season: int, opp_row) -> dict:
    """Game log, usage and matchup numbers behind one projection (explanations only)."""
    cur = rows[rows.season == season].sort_values("week", ascending=False)
    vals = _value(cur, st)
    prev = rows[rows.season == season - 1]
    d = {"gp": len(cur), "season_avg": float(vals.mean()) if len(cur) else np.nan,
         "prev_gp": len(prev), "prev_avg": float(_value(prev, st).mean()) if len(prev) else np.nan,
         "log": "|".join(f"{v:g}:{o}" for v, o in zip(vals.head(6).round(1), cur.opponent_team.head(6)))}
    n = max(1, len(cur))
    if st in ("rec_yds", "receptions"):
        d["tgt_pg"] = float(cur.targets.fillna(0).sum() / n)
        d["tshare"] = float(cur.target_share.mean()) if "target_share" in cur and cur.target_share.notna().any() else np.nan
        d["ay_pg"] = float(cur.receiving_air_yards.fillna(0).sum() / n)
        tg = cur.targets.fillna(0).sum()
        d["catch"] = float(cur.receptions.fillna(0).sum() / tg) if tg else np.nan
    elif st == "rush_yds":
        car = cur.carries.fillna(0).sum()
        d["car_pg"] = float(car / n)
        d["ypc"] = float(cur.rushing_yards.fillna(0).sum() / car) if car else np.nan
    elif st in ("pass_yds", "pass_tds"):
        att = cur.attempts.fillna(0).sum()
        d["att_pg"] = float(att / n)
        d["ypa"] = float(cur.passing_yards.fillna(0).sum() / att) if att else np.nan
    if opp_row is not None:
        d["opp_pg"], d["opp_rank"], d["opp_n"] = opp_row
    return d


def _why_td_detail(rows: pd.DataFrame, pid, season: int, rz) -> dict:
    """Touchdown history and red-zone usage this season (explanations only)."""
    cur = rows[rows.season == season]
    d = {"rush_td": float(cur.rushing_tds.fillna(0).sum()), "rec_td": float(cur.receiving_tds.fillna(0).sum())}
    if rz is not None and len(rz):
        r = rz[(rz.player_id == pid) & (rz.season == season)]
        n = max(1, len(cur))
        d["rz_pg"] = float((r.rz_carries.fillna(0) + r.rz_targets.fillna(0)).sum() / n)
        d["gl_pg"] = float((r.gl_carries.fillna(0) + r.gl_targets.fillna(0)).sum() / n)
    return d


def _opponent_factors(stats: pd.DataFrame, season: int, week: int) -> dict:
    """(opponent, position, stat) -> how much more/less that defense allows than average, last 8 games."""
    s = stats[(stats.season == season) | ((stats.season == season - 1) & (stats.week >= 12))]
    out = {}
    for stat in ("rush_yds", "rec_yds", "receptions", "pass_yds", "pass_tds"):
        col = STATS[stat][0]
        g = s.groupby(["opponent_team", "position", "season", "week"])[col].sum().reset_index()
        g = g.sort_values(["season", "week"]).groupby(["opponent_team", "position"]).tail(8)
        per = g.groupby(["opponent_team", "position"])[col].mean()
        lg = per.groupby(level="position").mean()
        for (opp, pos), v in per.items():
            if lg.get(pos, 0) > 0:
                out[(opp, pos, stat)] = float(np.clip(1 + OPP_SHRINK * (v / lg[pos] - 1), 0.85, 1.15))
    return out


def _redistribution(stats, season, week, team, on_team, eligible) -> dict:
    """Boost for the remaining players when teammates who were getting targets/carries are out."""
    r = stats[(stats.season == season) & (stats.team == team) & (stats.week < week)]
    recent = sorted(r.week.unique())[-3:]
    r = r[r.week.isin(recent)]
    out = {}
    for key, col in (("rec", "targets"), ("rush", "carries")):
        tot = r[col].fillna(0).sum()
        gone = r[~r.player_id.isin(eligible) & r.player_id.isin(on_team) & (r.position != "QB")][col].fillna(0).sum()
        share = gone / tot if tot > 0 else 0.0
        out[key] = float(np.clip(1 + REDIS_WEIGHT * share / max(0.2, 1 - share), 1.0, 1.35))
    return out


def _volume_factors(tend, team: str, opp: str) -> dict:
    """Plays and pass/run mix expected this game (team tendency met by opponent's) vs. the team's norm."""
    if tend is None or team not in tend.index or opp not in tend.index:
        return {"pass": 1.0, "rush": 1.0}
    t, o = tend.loc[team], tend.loc[opp]
    plays = (t.plays + o.plays_faced) / 2
    pr = (t.pass_rate + o.pass_rate_faced) / 2
    vol = plays / t.plays
    return {"pass": float(np.clip(vol * pr / t.pass_rate, 0.9, 1.1)),
            "rush": float(np.clip(vol * (1 - pr) / (1 - t.pass_rate), 0.9, 1.1))}


def _usage_est(rows: pd.DataFrame, stat: str, season: int) -> float:
    """Receiving yards / receptions implied by targets, air yards and catch rate (steadier than raw totals)."""
    r = rows.sort_values(["season", "week"], ascending=False)
    w = RECENCY ** np.arange(len(r)) * np.where(r.season < season, PRIOR_SEASON_W, 1.0)
    tgt = r.targets.fillna(0).to_numpy(float)
    if stat == "rec_yds":
        x = AIR_YDS_COEF * r.receiving_air_yards.fillna(0).clip(lower=0).to_numpy(float) + TGT_YDS_COEF * tgt
        return float((w * x).sum() / w.sum())
    catch = (r.receptions.fillna(0).sum() + LG_CATCH * CATCH_SHRINK) / (tgt.sum() + CATCH_SHRINK)
    return float((w * tgt).sum() / w.sum() * catch)


def _pressure(tm, season: int, week: int) -> dict:
    """(team, opponent) -> passing multiplier from the team's pressure allowed vs. the opponent's pressure generated."""
    if tm is None or "pressured" not in tm:
        return {}
    t = tm[(tm.season < season) | ((tm.season == season) & (tm.week < week))].sort_values(["season", "week"])
    off = t.groupby("team").tail(8).groupby("team").apply(lambda g: g.pressured.sum() / max(1, g.dropbacks.sum()), include_groups=False)
    dfn = t.groupby("team").tail(8).groupby("team").apply(lambda g: g.pressures.sum() / max(1, g.dropbacks_faced.sum()), include_groups=False)
    lg = float(t.pressured.sum() / max(1, t.dropbacks.sum()))
    out = {}
    for a in off.index:
        for b in dfn.index:
            exp = (off[a] + dfn[b]) / 2
            out[(a, b)] = float(np.clip(1 - PRESSURE_W * (exp / lg - 1), 0.9, 1.1))
    return out


def _route_factor(rows: pd.DataFrame, routes_t, pid: str, pos: str, season: int) -> float:
    """Targets projected as (targets per route run) x (expected routes), relative to the plain weighted
    targets average the projection already uses. >1 when his routes recently went up (role grew)."""
    if routes_t is None:
        return 1.0
    r = rows[["season", "week", "targets"]].merge(routes_t[routes_t.player_id == pid], on=["season", "week"], how="inner")
    r = r.sort_values(["season", "week"], ascending=False)
    if len(r) < 2:
        return 1.0
    tg = r.targets.fillna(0).to_numpy(float)
    rr = r.r.to_numpy(float)
    w = RECENCY ** np.arange(len(r)) * np.where(r.season < season, PRIOR_SEASON_W, 1.0)
    w2 = ROUTE_RECENCY ** np.arange(len(r)) * np.where(r.season < season, PRIOR_SEASON_W, 1.0)
    prior = TPRR_PRIOR.get(pos, 0.18)
    tprr = ((w * tg).sum() + prior * TPRR_SHRINK) / ((w * rr).sum() + TPRR_SHRINK)
    routes_hat = (w2 * rr).sum() / w2.sum()
    tgt_plain = (w * tg).sum() / w.sum()
    if tgt_plain <= 0.3:
        return 1.0
    ratio = float(np.clip(tprr * routes_hat / tgt_plain, 0.6, 1.5))
    return 1 + ROUTE_W * (ratio - 1)
