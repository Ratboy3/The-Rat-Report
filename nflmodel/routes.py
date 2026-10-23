"""Routes run per player-game, for targets-per-route-run receiving projections.

  python run.py build-routes      # -> data/routes.csv

Two sources:
  exact  - nflverse participation data: the player was on the field on a QB dropback (a route, or for some
           backs and tight ends a pass-protection snap). Published after each season, so last season only.
  proxy  - snap counts x team dropbacks (offensive snap share applied to the team's dropbacks). Available
           every week of the current season, so this is what the live model can use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
FILE = DATA / "routes.csv"
PART_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"


def _exact(season: int) -> pd.DataFrame:
    part_path = RAW / f"part_{season}.csv"
    part = pd.read_csv(part_path if part_path.exists() else PART_URL.format(season=season),
                       usecols=["nflverse_game_id", "play_id", "possession_team", "offense_players"], low_memory=False)
    pbp_path = RAW / f"pbp_{season}.csv.gz"
    pbp = pd.read_csv(pbp_path if pbp_path.exists() else PBP_URL.format(season=season),
                      usecols=["game_id", "play_id", "season_type", "week", "qb_dropback", "posteam"], low_memory=False)
    pbp = pbp[(pbp.season_type == "REG") & (pbp.qb_dropback == 1)]
    d = part.merge(pbp, left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"])
    d = d[d.offense_players.notna()]
    d = d.assign(player_id=d.offense_players.str.split(";")).explode("player_id")
    g = d.groupby(["game_id", "week", "posteam", "player_id"]).size().rename("routes").reset_index()
    g["season"] = season
    return g.rename(columns={"posteam": "team"})


def _proxy(season: int) -> pd.DataFrame:
    from . import injuries, pbp_extra
    from . import player_proj as PP
    snaps = injuries._load_snaps(season, False)
    roster = PP._load("roster", season, False)
    _, tm = pbp_extra.load()
    if snaps is None or roster is None or tm is None:
        return pd.DataFrame()
    pfr = roster.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    s = snaps[(snaps.game_type == "REG") & (snaps.offense_pct > 0)].copy()
    s["player_id"] = s.pfr_player_id.map(pfr)
    t = tm[tm.season == season][["week", "team", "dropbacks"]]
    s = s.merge(t, on=["week", "team"], how="inner")
    s["routes_proxy"] = s.offense_pct * s.dropbacks
    return s.dropna(subset=["player_id"])[["season", "week", "team", "player_id", "routes_proxy"]]


def build(seasons: list[int]) -> pd.DataFrame:
    parts = []
    for s in seasons:
        px = _proxy(s)
        try:
            ex = _exact(s)
            print(f"  {s}: participation {len(ex):,} player-games")
        except Exception as e:
            ex = pd.DataFrame(columns=["season", "week", "team", "player_id", "routes"])
            print(f"  {s}: no participation yet ({type(e).__name__})")
        m = px.merge(ex[["season", "week", "team", "player_id", "routes"]], on=["season", "week", "team", "player_id"], how="outer")
        parts.append(m)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(FILE, index=False)
    print(f"Wrote {FILE.name} ({len(out):,} player-games)")
    return out


_CACHE: dict = {}


def load() -> pd.DataFrame | None:
    if not FILE.exists():
        return None
    k = FILE.stat().st_mtime
    if k not in _CACHE:
        _CACHE.clear()
        _CACHE[k] = pd.read_csv(FILE)
    return _CACHE[k]
