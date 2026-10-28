"""Quarterback quality: replaces the yes/no backup-QB flag with a rating for each quarterback.

Rating = EPA per dropback (passes, sacks and scrambles), recent games weighted most, last season's
games at half weight, shrunk toward a below-average prior so a QB with little film is assumed to be a
typical backup until he shows otherwise.

When a team starts someone other than its usual QB (most common starter of its last 6 games), including
a regular starter returning from injury, the spread gets
  points = (new QB rating - usual QB rating) x dropbacks per game x QB_PTS_SCALE
in place of the old yes/no flag. Tested on 112 QB-change games in 2025-26: average miss on the final margin
11.89 -> 11.62 points with the ratings. The total kept the yes/no flag, which tested better there (10.94 vs 11.17).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
QB_FILE = ROOT / "data" / "pbp_qb_games.csv"

PRIOR_EPA = -0.08        # what an unknown QB is assumed to be (league average is about +0.05)
PRIOR_DROPBACKS = 150    # how much film it takes to move halfway off the prior
RECENCY = 0.9            # each older game counts 10% less
PRIOR_SEASON_W = 0.5
DROPBACKS_PER_GAME = 36
QB_PTS_SCALE = 1.0       # set from the QB-change backtest (see qb_backtest in run.py)
MAX_PTS = 12.0


def _load() -> pd.DataFrame | None:
    return pd.read_csv(QB_FILE) if QB_FILE.exists() else None


def rating(qbg: pd.DataFrame, qb_id: str, season: int, week: int) -> tuple[float, float]:
    """(EPA per dropback, dropbacks of evidence) using only games before (season, week)."""
    g = qbg[(qbg.qb_id == qb_id) & ((qbg.season < season) | ((qbg.season == season) & (qbg.week < week)))]
    g = g[g.season >= season - 1].sort_values(["season", "week"], ascending=False)
    if g.empty:
        return PRIOR_EPA, 0.0
    w = RECENCY ** np.arange(len(g)) * np.where(g.season < season, PRIOR_SEASON_W, 1.0)
    db = (w * g.dropbacks).sum()
    epa = (w * g.epa).sum()
    return float((epa + PRIOR_EPA * PRIOR_DROPBACKS) / (db + PRIOR_DROPBACKS)), float(db)


def _ids_by_name(season: int) -> dict:
    """QB full name -> gsis id, from rosters (current season first) and the schedule."""
    from .player_proj import _load            # downloads the roster if this run hasn't yet
    out = {}
    for yr in (season, season - 1):
        r = _load("roster", yr, False)
        if r is not None:
            r = r[r.position == "QB"].dropna(subset=["gsis_id", "full_name"])
            for n, i in zip(r.full_name, r.gsis_id):
                out.setdefault(n, i)
    return out


def attach(df: pd.DataFrame, games: pd.DataFrame, season: int, week: int) -> tuple[pd.DataFrame, dict]:
    """Add QB point adjustments (folded into home/away_inj_off) for a week's games."""
    qbg = _load()
    info = {}
    if qbg is None or df.empty:
        return df.assign(home_qb_pts=np.nan, away_qb_pts=np.nan), info
    ids = _ids_by_name(season)
    df = df.copy()
    for side in ("home", "away"):
        df[f"{side}_qb_pts"] = np.nan
    for i, g in df.iterrows():
        for side in ("home", "away"):
            team = g[f"{side}_team"]
            # usual starter: the QB with the most dropbacks in each of the team's previous 6 games (play-by-play,
            # which is reliable; the schedule's QB names are sometimes wrong)
            tg = qbg[(qbg.posteam == team) & ((qbg.season < season) | ((qbg.season == season) & (qbg.week < week)))]
            if tg.empty:
                continue
            starters = (tg.sort_values("dropbacks").groupby(["season", "week"]).tail(1)
                        .sort_values(["season", "week"]).tail(6))
            counts = starters.qb_id.value_counts()
            top = counts[counts == counts.max()].index
            usual_id = starters[starters.qb_id.isin(top)].qb_id.iloc[-1]      # ties: most recent
            usual_name = starters[starters.qb_id == usual_id].qb_name.iloc[-1]
            new_name = g[f"{side}_qb_name"]
            new_id = ids.get(new_name)
            if new_id is None:   # try schedule history (e.g. traded QB)
                m = pd.concat([games[["home_qb_name", "home_qb_id"]].set_axis(["n", "i"], axis=1),
                               games[["away_qb_name", "away_qb_id"]].set_axis(["n", "i"], axis=1)]).dropna()
                m = m[m.n == new_name]
                new_id = m.i.iloc[-1] if len(m) else None
            if (new_id is not None and new_id == usual_id) or (new_id is None and new_name == usual_name):
                continue                                   # the usual starter plays: nothing to adjust
            r_new, db_new = rating(qbg, new_id, season, week) if new_id else (PRIOR_EPA, 0.0)
            r_old, db_old = rating(qbg, usual_id, season, week)
            pts = float(np.clip((r_new - r_old) * DROPBACKS_PER_GAME * QB_PTS_SCALE, -MAX_PTS, MAX_PTS))
            df.at[i, f"{side}_qb_pts"] = pts      # used for the spread; the total keeps the yes/no flag (tested better)
            info[team] = dict(new=new_name, new_rating=r_new, new_db=db_new, usual=usual_name, usual_rating=r_old, pts=pts)
    return df, info


OUT_STATUSES = {"Out", "Doubtful"}
ROSTER_OUT = {"RES", "INA", "PUP", "NON", "SUS", "CUT", "RET"}   # IR / reserve / inactive / not with the team


def status(df: pd.DataFrame, season: int, week: int) -> dict:
    """team -> who is starting at QB and why, checked against the injury report and roster (Oct 28, 2026).

    usual = most starts in the team's last 6 games (play-by-play dropbacks); opener = this season's Week 1 starter.
    kind: 'starter'   the usual starter plays
          'returning' the opening-day starter is back after someone else's run (not a backup)
          'backup'    someone else starts and the usual starter is out / on IR / gone (injury report, roster)
          'unclear'   someone else is listed but the report and roster don't show the usual starter out: confirm"""
    qbg = _load()
    out = {}
    if qbg is None or df.empty:
        return out
    from .player_proj import _load as load_nfl
    ids = _ids_by_name(season)
    inj = load_nfl("injuries", season, False)
    ros = load_nfl("roster", season, False)
    inj_w = inj[(inj.season == season) & (inj.week == week)] if inj is not None and len(inj) else None
    ros_w = None
    if ros is not None and len(ros) and "week" in ros:
        ros_w = ros[ros.week == min(week, ros.week.max())]
    for g in df.itertuples():
        for side in ("home", "away"):
            team, name = getattr(g, f"{side}_team"), getattr(g, f"{side}_qb_name")
            tg = qbg[(qbg.posteam == team) & ((qbg.season < season) | ((qbg.season == season) & (qbg.week < week)))]
            if tg.empty or not isinstance(name, str):
                continue
            st = tg.sort_values("dropbacks").groupby(["season", "week"]).tail(1).sort_values(["season", "week"])
            last6 = st.tail(6)
            c6 = last6.qb_id.value_counts()
            usual_id = last6[last6.qb_id.isin(c6[c6 == c6.max()].index)].qb_id.iloc[-1]
            usual_name = last6[last6.qb_id == usual_id].qb_name.iloc[-1]
            this = st[st.season == season]
            opener_id = this.qb_id.iloc[0] if len(this) else None
            new_id = ids.get(name)
            is_ = lambda qid, nm: (new_id == qid) if new_id else (name.split()[-1] == str(nm).split()[-1])
            if is_(usual_id, usual_name):
                out[team] = dict(kind="starter", qb=name, text=f"{name} (usual starter)")
                continue
            if opener_id is not None and is_(opener_id, this.qb_name.iloc[0]):
                n_bk = int((last6.qb_id != opener_id).sum())
                out[team] = dict(kind="returning", qb=name, other=usual_name, games=n_bk,
                                 text=f"{name} returns as the starter ({usual_name} started {n_bk} of the last 6)")
                continue
            why = None                                   # someone else is listed: is the usual starter out?
            if inj_w is not None:
                r = inj_w[inj_w.gsis_id == usual_id]
                if len(r) and str(r.report_status.iloc[0]) in OUT_STATUSES:
                    inj_txt = r.report_primary_injury.iloc[0]
                    why = str(r.report_status.iloc[0]) + (f", {inj_txt}" if isinstance(inj_txt, str) else "")
            if why is None and ros_w is not None:
                r = ros_w[ros_w.gsis_id == usual_id]
                if len(r) and str(r.status.iloc[0]) in ROSTER_OUT:
                    why = {"RES": "injured reserve", "INA": "inactive", "PUP": "PUP list", "SUS": "suspended",
                           "CUT": "released", "RET": "retired", "NON": "non-football injury list"}.get(str(r.status.iloc[0]), str(r.status.iloc[0]))
                elif len(r) == 0 or r.team.iloc[0] != team:
                    why = "no longer on the roster"
            if why:
                out[team] = dict(kind="backup", qb=name, other=usual_name, text=f"{name} starts for {usual_name} ({why})")
            else:
                out[team] = dict(kind="unclear", qb=name, other=usual_name,
                                 text=f"{name} is listed, but {usual_name} isn't shown out on the injury report or roster: confirm")
    return out


def apply_status(df: pd.DataFrame, st: dict) -> pd.DataFrame:
    """Set the QB-change flags from the injury-report check: a backup is a change, the established or returning
    starter is not (for the total and props). 'unclear' keeps the schedule-based flags and is shown as a warning."""
    df = df.copy()
    for i, g in df.iterrows():
        for side in ("home", "away"):
            s = st.get(g[f"{side}_team"])
            if not s:
                continue
            if s["kind"] == "backup":
                df.at[i, f"{side}_qb_change"] = 1
                df.at[i, f"{side}_qb_change_t"] = 1
            elif s["kind"] in ("starter", "returning"):
                df.at[i, f"{side}_qb_change_t"] = 0
                if s["kind"] == "starter":
                    df.at[i, f"{side}_qb_change"] = 0
    return df
