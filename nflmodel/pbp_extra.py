"""Extra play-by-play summaries for player props and team tendencies.

  data/pbp_player_games.csv  one row per player per game: red-zone and goal-line carries/targets
  data/pbp_team_games.csv    one row per team per game: offensive plays, neutral-situation pace and
                             pass rate over expected, 4th-down aggressiveness, plays faced on defense

Built from nflverse play-by-play (downloaded to data/raw/, ~20 MB a season).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PLAYER_FILE = ROOT / "data" / "pbp_player_games.csv"
TEAM_FILE = ROOT / "data" / "pbp_team_games.csv"
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "yardline_100", "down",
        "ydstogo", "wp", "half_seconds_remaining", "game_seconds_remaining", "drive", "pass", "rush", "xpass",
        "rusher_player_id", "receiver_player_id", "qb_kneel", "qb_spike", "two_point_attempt",
        "qb_dropback", "qb_epa", "passer_player_id", "passer_player_name", "rusher_player_name", "qb_hit", "sack"]
QB_FILE = ROOT / "data" / "pbp_qb_games.csv"


def _download(season: int, refresh: bool) -> Path | None:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"pbp_{season}.csv.gz"
    if refresh or not p.exists():
        try:
            import urllib.request
            urllib.request.urlretrieve(URL.format(season=season), p)
        except Exception as e:
            print(f"  could not download play-by-play {season} ({e})")
            if not p.exists():
                return None
    return p


def _season(src: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(src, usecols=COLS, low_memory=False)
    d = d[(d.season_type == "REG") & d.posteam.notna()]
    for c in ("posteam", "defteam"):
        d[c] = d[c].replace(C.TEAM_RENAMES)
    plays = d[d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1) & (d.two_point_attempt != 1)].copy()

    # ---- players: red-zone (inside the 20) and goal-line (inside the 5) opportunities
    rz = plays.yardline_100 <= 20
    gl = plays.yardline_100 <= 5
    car = plays[plays.rush == 1].assign(rz=rz, gl=gl).groupby(["season", "week", "game_id", "rusher_player_id"]).agg(
        rz_carries=("rz", "sum"), gl_carries=("gl", "sum")).reset_index().rename(columns={"rusher_player_id": "player_id"})
    tgt = plays[plays["pass"] == 1].assign(rz=rz, gl=gl).groupby(["season", "week", "game_id", "receiver_player_id"]).agg(
        rz_targets=("rz", "sum"), gl_targets=("gl", "sum")).reset_index().rename(columns={"receiver_player_id": "player_id"})
    pl = car.merge(tgt, on=["season", "week", "game_id", "player_id"], how="outer").fillna(0)

    # ---- teams: volume, neutral pace and pass rate over expected, 4th-down aggressiveness
    neutral = plays[(plays.wp.between(0.2, 0.8)) & (plays.half_seconds_remaining > 120)].copy()
    neutral = neutral.sort_values(["game_id", "drive", "game_seconds_remaining"], ascending=[True, True, False])
    neutral["gap"] = -neutral.groupby(["game_id", "drive"]).game_seconds_remaining.diff()
    neutral.loc[(neutral.gap <= 0) | (neutral.gap > 60), "gap"] = np.nan
    neutral["proe"] = neutral["pass"] - neutral.xpass
    fourth = d[(d.down == 4) & (d.ydstogo <= 3) & d.yardline_100.between(30, 75)].copy()
    fourth["go"] = fourth.play_type.isin(["pass", "run"]).astype(float)

    off = plays.groupby(["season", "week", "game_id", "posteam"]).agg(
        plays=("play_type", "size"), pass_rate=("pass", "mean"), rz_plays=("yardline_100", lambda s: (s <= 20).sum())).reset_index()
    neu = neutral.groupby(["season", "week", "game_id", "posteam"]).agg(
        sec_per_play=("gap", "median"), proe=("proe", "mean")).reset_index()
    go = fourth.groupby(["season", "week", "game_id", "posteam"]).agg(fourth_tries=("go", "size"), fourth_go=("go", "sum")).reset_index()
    dfn = plays.groupby(["season", "week", "game_id", "defteam"]).agg(
        plays_faced=("play_type", "size"), pass_rate_faced=("pass", "mean")).reset_index().rename(columns={"defteam": "posteam"})
    tm = off.merge(neu, how="left").merge(go, how="left").merge(dfn, how="left").rename(columns={"posteam": "team"})
    tm[["fourth_tries", "fourth_go"]] = tm[["fourth_tries", "fourth_go"]].fillna(0)

    # ---- pass rush vs. protection: sacks + QB hits per dropback, allowed (offense) and generated (defense)
    db = d[(d.qb_dropback == 1) & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
    db["pressure"] = ((db.sack.fillna(0) > 0) | (db.qb_hit.fillna(0) > 0)).astype(float)
    pa = db.groupby(["season", "week", "game_id", "posteam"]).agg(dropbacks=("pressure", "size"), pressured=("pressure", "sum")).reset_index()
    pg = db.groupby(["season", "week", "game_id", "defteam"]).agg(dropbacks_faced=("pressure", "size"), pressures=("pressure", "sum")).reset_index()
    tm = tm.merge(pa.rename(columns={"posteam": "team"}), how="left").merge(pg.rename(columns={"defteam": "team"}), how="left")

    # ---- quarterbacks: EPA per dropback (scrambles credited to the QB)
    db["qb_id"] = db.passer_player_id.fillna(db.rusher_player_id)
    db["qb_name"] = db.passer_player_name.fillna(db.rusher_player_name)
    qb = db.dropna(subset=["qb_id"]).groupby(["season", "week", "game_id", "posteam", "qb_id"]).agg(
        qb_name=("qb_name", "first"), dropbacks=("qb_epa", "size"), epa=("qb_epa", "sum")).reset_index()
    tm.attrs["qb"] = qb
    return pl, tm


def build(seasons: list[int], refresh_current: bool = True) -> None:
    pls, tms, qbs = [], [], []
    for i, s in enumerate(seasons):
        src = _download(s, refresh_current and i == len(seasons) - 1)
        if src is None:
            continue
        pl, tm = _season(src)
        pls.append(pl)
        qbs.append(tm.attrs.pop("qb"))
        tms.append(tm)
    if pls:
        pd.concat(pls).to_csv(PLAYER_FILE, index=False)
        pd.concat(tms).to_csv(TEAM_FILE, index=False)
        pd.concat(qbs).to_csv(QB_FILE, index=False)
        print(f"Wrote {PLAYER_FILE.name}, {TEAM_FILE.name} and {QB_FILE.name}")


def load() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    pl = pd.read_csv(PLAYER_FILE) if PLAYER_FILE.exists() else None
    tm = pd.read_csv(TEAM_FILE) if TEAM_FILE.exists() else None
    return pl, tm


def tendencies(tm: pd.DataFrame, season: int, week: int, n: int = 6) -> pd.DataFrame:
    """Team tendencies over its last n games before this week (recent-weighted), vs league average."""
    t = tm[(tm.season < season) | ((tm.season == season) & (tm.week < week))].sort_values(["season", "week"])
    rows = []
    for team, g in t.groupby("team"):
        g = g.tail(n)
        w = 0.85 ** np.arange(len(g))[::-1]
        avg = lambda c: float(np.nansum(g[c] * w) / np.sum(w[~np.isnan(g[c].to_numpy(float))])) if g[c].notna().any() else np.nan
        rows.append(dict(team=team, plays=avg("plays"), plays_faced=avg("plays_faced"), pass_rate=avg("pass_rate"),
                         pass_rate_faced=avg("pass_rate_faced"), sec_per_play=avg("sec_per_play"), proe=avg("proe"),
                         go_rate=float(g.fourth_go.sum() / g.fourth_tries.sum()) if g.fourth_tries.sum() >= 3 else np.nan))
    out = pd.DataFrame(rows)
    for c in ("plays", "plays_faced", "pass_rate", "pass_rate_faced", "sec_per_play", "proe", "go_rate"):
        out[f"{c}_lg"] = out[c].mean()
    return out
