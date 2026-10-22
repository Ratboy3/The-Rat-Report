"""Coverage matchups from last season's charting (nflverse participation data: man vs. zone on each pass).

The current season's charting isn't released yet, so this uses the previous season:
  - each defense's man-coverage rate on dropbacks
  - each receiver's yards per target against man and against zone (shrunk toward league average)
A receiver who does much better against zone, facing a heavy-zone defense, gets a small bump, and vice versa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
SHRINK_TARGETS = 30


def _get(url: str, path: Path) -> Path | None:
    if not path.exists():
        try:
            import urllib.request
            RAW.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"  could not download {path.name} ({e})")
            return None
    return path


def tables(season: int) -> tuple[dict, dict, float]:
    """For predicting `season`: ({defense: man rate}, {receiver gsis id: (ypt_man, ypt_zone)}, league man rate),
    built from season - 1."""
    prev = season - 1
    cache = ROOT / "data" / f"coverage_{prev}.csv"
    dcache = ROOT / "data" / f"def_man_{prev}.csv"
    if not cache.exists() or not dcache.exists():
        part = _get(URL.format(season=prev), RAW / f"part_{prev}.csv")
        pbp = _get(PBP_URL.format(season=prev), RAW / f"pbp_{prev}.csv.gz")
        if part is None or pbp is None:
            return {}, {}, 0.3
        pa = pd.read_csv(part, usecols=["nflverse_game_id", "play_id", "defense_man_zone_type"], low_memory=False)
        pb = pd.read_csv(pbp, usecols=["game_id", "play_id", "season_type", "defteam", "receiver_player_id", "receiving_yards",
                                       "pass_attempt", "sack", "qb_dropback"], low_memory=False)
        pb = pb[(pb.season_type == "REG") & (pb.qb_dropback == 1)]
        m = pb.merge(pa.rename(columns={"nflverse_game_id": "game_id"}), on=["game_id", "play_id"], how="inner")
        m = m[m.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
        m["man"] = (m.defense_man_zone_type == "MAN_COVERAGE").astype(float)
        m["defteam"] = m.defteam.replace(C.TEAM_RENAMES)
        m.groupby("defteam").man.mean().rename("man_rate").to_csv(dcache)
        t = m.dropna(subset=["receiver_player_id"]).copy()
        t["yds"] = t.receiving_yards.fillna(0)
        lg = t.groupby("man").yds.mean()
        g = t.groupby(["receiver_player_id", "man"]).yds.agg(["sum", "size"]).unstack(fill_value=0)
        out = pd.DataFrame(index=g.index)
        for k, col in ((1.0, "ypt_man"), (0.0, "ypt_zone")):
            s, n = g[("sum", k)], g[("size", k)]
            out[col] = (s + lg[k] * SHRINK_TARGETS) / (n + SHRINK_TARGETS)
        out.to_csv(cache)
    d = pd.read_csv(dcache, index_col=0).man_rate.to_dict()
    r = pd.read_csv(cache, index_col=0)
    return d, {i: (a, b) for i, a, b in zip(r.index, r.ypt_man, r.ypt_zone)}, float(np.mean(list(d.values())))


def factor(splits: dict, man: dict, lg_man: float, player_id: str, opp: str, weight: float = 0.5) -> float:
    """Receiving-yards multiplier for this receiver against this defense's coverage mix."""
    if player_id not in splits or opp not in man:
        return 1.0
    ym, yz = splits[player_id]
    vs_opp = man[opp] * ym + (1 - man[opp]) * yz
    vs_avg = lg_man * ym + (1 - lg_man) * yz
    return float(np.clip(1 + weight * (vs_opp / vs_avg - 1), 0.9, 1.1)) if vs_avg > 0 else 1.0
