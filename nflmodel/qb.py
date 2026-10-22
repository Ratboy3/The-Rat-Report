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
