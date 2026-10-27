"""DVOA-style unit efficiency from free nflverse play-by-play (built Oct 27, 2026).

DVOA itself is licensed by FTN. This builds the same kind of measure from public data, one row per team-game:
  - pass / rush splits of EPA per play and success rate, for every offense (and so every defense faced)
  - "DVOA success": a play succeeds if it gains 40% of the yards needed on 1st down, 60% on 2nd, or all of
    them on 3rd/4th (Football Outsiders' published baseline), overall and split pass / rush
Garbage time (win probability outside 5-95%) is left out, as in epa.py. Opponent adjustment happens when ratings
are fit (ratings.fit_play_ratings), the same ridge fit the EPA ratings use.

  python run.py build-units      # -> data/unit_team_games.csv (all seasons since 2006, a few minutes)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "unit_team_games.csv"
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "epa", "success", "wp",
        "pass", "rush", "down", "ydstogo", "yards_gained"]
METRICS = ["pass_epa", "rush_epa", "pass_sr", "rush_sr", "dsr", "pass_dsr", "rush_dsr"]


def condense(src) -> pd.DataFrame:
    d = pd.read_csv(src, usecols=COLS, low_memory=False)
    d = d[d.play_type.isin(["pass", "run"]) & d.epa.notna() & d.posteam.notna() & d.down.notna()]
    d = d[(d.wp >= C.EPA_WP_MIN) & (d.wp <= C.EPA_WP_MAX)]
    need = np.select([d.down == 1, d.down == 2], [0.4, 0.6], 1.0) * d.ydstogo
    d["dsucc"] = (d.yards_gained.fillna(0) >= need).astype(float)
    ps, rs = d["pass"] == 1, d["rush"] == 1
    for name, col, mask in (("pass_epa", "epa", ps), ("rush_epa", "epa", rs), ("pass_sr", "success", ps),
                            ("rush_sr", "success", rs), ("pass_dsr", "dsucc", ps), ("rush_dsr", "dsucc", rs)):
        d[name] = np.where(mask, d[col], np.nan)
    d["dsr"] = d.dsucc
    g = d.groupby(["game_id", "season", "week", "posteam", "defteam"], as_index=False).agg(
        pass_plays=("pass_epa", "count"), rush_plays=("rush_epa", "count"), **{m: (m, "mean") for m in METRICS})
    g = g.rename(columns={"posteam": "team", "defteam": "opp"})
    for c in ("team", "opp"):
        g[c] = g[c].replace(C.TEAM_RENAMES)
    return g


def build(first: int = 2006, last: int | None = None, only: list[int] | None = None) -> pd.DataFrame:
    import datetime
    last = last or datetime.date.today().year
    seasons = only or list(range(first, last + 1))
    parts = []
    for s in seasons:
        try:
            parts.append(condense(URL.format(season=s)))
            print(f"  {s}: done", flush=True)
        except Exception as e:
            print(f"  {s}: skipped ({e})")
    new = pd.concat(parts, ignore_index=True)
    if only and FILE.exists():
        old = pd.read_csv(FILE)
        new = pd.concat([old[~old.season.isin(only)], new], ignore_index=True)
    new = new.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    FILE.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(FILE, index=False, float_format="%.5f")
    print(f"Wrote {FILE.name} ({len(new):,} team-games)")
    return new


def load() -> pd.DataFrame | None:
    return pd.read_csv(FILE) if FILE.exists() else None
