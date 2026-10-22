"""Play-by-play efficiency: EPA per play and success rate for every team in every game.

Raw play-by-play is ~20 MB per season, so it is condensed once into
data/epa_team_games.csv (one row per team per game). `refresh` re-downloads
only the current season and rebuilds that season's rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
PBP_DIR = ROOT / "data" / "pbp"
EPA_FILE = ROOT / "data" / "epa_team_games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
COLS = ["game_id", "season", "week", "posteam", "defteam", "home_team", "play_type", "epa", "success",
        "wp", "pass", "rush"]


def condense_season(src) -> pd.DataFrame:
    d = pd.read_csv(src, usecols=COLS, low_memory=False)
    d = d[d.play_type.isin(["pass", "run"]) & d.epa.notna() & d.posteam.notna()]
    # Leave out garbage time, where teams play for the clock, not for points
    d = d[(d.wp >= C.EPA_WP_MIN) & (d.wp <= C.EPA_WP_MAX)]
    d["pass_epa"] = np.where(d["pass"] == 1, d.epa, np.nan)
    d["rush_epa"] = np.where(d["rush"] == 1, d.epa, np.nan)
    g = d.groupby(["game_id", "season", "week", "posteam", "defteam", "home_team"], as_index=False).agg(
        plays=("epa", "size"), epa=("epa", "mean"), sr=("success", "mean"),
        pass_epa=("pass_epa", "mean"), rush_epa=("rush_epa", "mean"))
    g = g.rename(columns={"posteam": "team", "defteam": "opp"})
    for c in ("team", "opp", "home_team"):
        g[c] = g[c].replace(C.TEAM_RENAMES)
    g["is_home"] = (g.team == g.home_team).astype(int)
    return g.drop(columns="home_team")


def build(seasons=None) -> pd.DataFrame:
    """Condense every local play-by-play file into EPA_FILE."""
    files = sorted(PBP_DIR.glob("pbp_*.csv.gz"))
    if seasons:
        files = [f for f in files if int(f.stem.split("_")[1].split(".")[0]) in seasons]
    parts = []
    for f in files:
        print(f"  condensing {f.name}")
        parts.append(condense_season(f))
    new = pd.concat(parts, ignore_index=True)
    if seasons and EPA_FILE.exists():
        old = pd.read_csv(EPA_FILE)
        new = pd.concat([old[~old.season.isin(seasons)], new], ignore_index=True)
    new = new.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    new.to_csv(EPA_FILE, index=False, float_format="%.5f")
    return new


def build_from_web(first: int = 2006, last: int | None = None) -> None:
    """Download and condense every season straight from nflverse (~400 MB streamed, a few minutes)."""
    import datetime
    last = last or datetime.date.today().year
    parts = []
    for season in range(first, last + 1):
        print(f"  downloading and condensing {season}")
        try:
            parts.append(condense_season(PBP_URL.format(season=season)))
        except Exception as e:
            print(f"    skipped {season}: {e}")
    out = pd.concat(parts, ignore_index=True).sort_values(["season", "week", "game_id", "team"])
    EPA_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(EPA_FILE, index=False, float_format="%.5f")
    print(f"Wrote {EPA_FILE} ({len(out):,} team-games)")


def refresh(season: int) -> None:
    """Re-download one season of play-by-play from nflverse and rebuild its rows."""
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    dest = PBP_DIR / f"pbp_{season}.csv.gz"
    try:
        df = condense_season(PBP_URL.format(season=season))
    except Exception as e:  # offline or blocked
        print(f"Play-by-play download failed ({e}); keeping saved EPA data")
        return
    old = pd.read_csv(EPA_FILE) if EPA_FILE.exists() else pd.DataFrame()
    out = pd.concat([old[old.season != season] if len(old) else old, df], ignore_index=True)
    out.sort_values(["season", "week", "game_id", "team"]).to_csv(EPA_FILE, index=False, float_format="%.5f")
    print(f"Updated EPA for {season} ({df.game_id.nunique()} games)")


def load() -> pd.DataFrame:
    if not EPA_FILE.exists():
        raise SystemExit("data/epa_team_games.csv missing. Run: python run.py build-epa")
    return pd.read_csv(EPA_FILE)
