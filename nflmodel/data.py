"""Load nflverse game data and build the per-game features the model uses."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "games.csv"


def load_games(path: str | os.PathLike | None = None, refresh: bool = False) -> pd.DataFrame:
    """Read games.csv from a path, the local cache, or nflverse on GitHub."""
    src = Path(path) if path else CACHE
    if refresh or not src.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            df = pd.read_csv(C.DATA_URL)
            df.to_csv(CACHE, index=False)
        except Exception as e:  # offline, blocked, etc.
            df = _git_fetch_games()
            if df is not None:
                df.to_csv(CACHE, index=False)
            elif not CACHE.exists():
                raise SystemExit(f"Could not download {C.DATA_URL}: {e}")
            else:
                print(f"Download failed ({e}); using the saved copy in {CACHE}")
                df = pd.read_csv(CACHE)
    else:
        df = pd.read_csv(src)
    return prepare(df)


def _git_fetch_games() -> pd.DataFrame | None:
    """Fallback for networks that block raw file downloads but allow git: sparse-clone nflverse/nfldata."""
    import shutil
    import subprocess
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/nflverse/nfldata", str(tmp / "nfldata")],
                       check=True, capture_output=True, timeout=300)
        subprocess.run(["git", "-C", str(tmp / "nfldata"), "sparse-checkout", "set", "--no-cone", "/data/games.csv"],
                       check=True, capture_output=True, timeout=300)
        print("Downloaded games.csv via git")
        return pd.read_csv(tmp / "nfldata" / "data" / "games.csv")
    except Exception as e:
        print(f"git download failed too ({e})")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.season >= C.FIRST_SEASON].copy()
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(C.TEAM_RENAMES)
    df["date"] = pd.to_datetime(df["gameday"])
    df["played"] = df["result"].notna()
    df["neutral"] = (df["location"] == "Neutral").astype(int)
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    df["dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["playoff"] = (df["game_type"] != "REG").astype(int)

    # Market numbers, home perspective. nflverse spread_line > 0 means home favored.
    df["mkt_margin"] = df["spread_line"]
    df["mkt_total"] = df["total_line"]
    for c in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds"):
        df[c] = df[c].fillna(C.DEFAULT_ODDS)

    # Rest
    df["home_bye"] = (df["home_rest"] >= 13).astype(int)
    df["away_bye"] = (df["away_rest"] >= 13).astype(int)
    df["rest_diff"] = (df["home_rest"].clip(4, 14) - df["away_rest"].clip(4, 14)) / 7.0

    # Travel: West Coast teams in early Eastern kickoffs, plus long trips generally
    tz_h = df["home_team"].map(C.TEAM_TZ).fillna(0)
    tz_a = df["away_team"].map(C.TEAM_TZ).fillna(0)
    early = df["gametime"].fillna("13:00").str.slice(0, 2).astype(int) < 14
    not_neutral = df["neutral"] == 0
    df["away_west_early"] = ((tz_a - tz_h >= 2) & early & not_neutral).astype(int)
    df["long_trip"] = (((tz_a - tz_h).abs() >= 2) & not_neutral).astype(int)

    # Weather (outdoor only). Missing values become neutral defaults.
    outdoor = df["dome"] == 0
    wind = df["wind"].fillna(6.0)
    temp = df["temp"].fillna(60.0)
    df["wind_excess"] = np.where(outdoor, (wind - 10).clip(lower=0), 0.0)
    df["cold"] = np.where(outdoor, (temp < 32).astype(int), 0)
    df["precip"] = 0

    add_qb_flags(df)
    df["home_adj"] = 0.0
    df["away_adj"] = 0.0
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def add_qb_flags(df: pd.DataFrame) -> None:
    """Flag games where a team starts someone other than its usual QB."""
    long = pd.concat([
        df[["game_id", "date", "home_team", "home_qb_name"]].set_axis(["game_id", "date", "team", "qb"], axis=1).assign(side="home"),
        df[["game_id", "date", "away_team", "away_qb_name"]].set_axis(["game_id", "date", "team", "qb"], axis=1).assign(side="away"),
    ]).sort_values("date")
    flags = {}
    for team, grp in long.groupby("team"):
        hist: list[str] = []
        for row in grp.itertuples():
            recent = [q for q in hist[-C.QB_LOOKBACK_GAMES:] if isinstance(q, str)]
            # most frequent recent starter; ties go to whoever started most recently
            usual = max(reversed(recent), key=recent.count) if recent else None
            change = int(usual is not None and isinstance(row.qb, str) and row.qb != usual)
            flags[(row.game_id, row.side)] = change
            hist.append(row.qb)
    df["home_qb_change"] = [flags.get((g, "home"), 0) for g in df.game_id]
    df["away_qb_change"] = [flags.get((g, "away"), 0) for g in df.game_id]


def apply_overrides(df: pd.DataFrame, path: str | os.PathLike | None) -> pd.DataFrame:
    """Merge user overrides: injuries (points), weather, and QB changes for upcoming games.

    CSV columns (all optional except game_id):
      game_id, home_adj, away_adj, wind, temp, precip, home_qb_change, away_qb_change,
      spread_line, total_line, home_moneyline, away_moneyline
    """
    if not path or not Path(path).exists():
        return df
    ov = pd.read_csv(path, comment="#")
    df = df.copy()
    for row in ov.to_dict("records"):
        m = df["game_id"] == row["game_id"]
        if not m.any():
            print(f"  override skipped, unknown game_id: {row['game_id']}")
            continue
        outdoor = df.loc[m, "dome"].iloc[0] == 0
        for k, v in row.items():
            if k == "game_id" or pd.isna(v):
                continue
            if k == "wind":
                df.loc[m, "wind_excess"] = max(0.0, v - 10) if outdoor else 0.0
            elif k == "temp":
                df.loc[m, "cold"] = int(v < 32) if outdoor else 0
            elif k == "precip":
                df.loc[m, "precip"] = int(v) if outdoor else 0
            elif k == "spread_line":
                df.loc[m, ["spread_line", "mkt_margin"]] = v
            elif k == "total_line":
                df.loc[m, ["total_line", "mkt_total"]] = v
            else:
                df.loc[m, k] = v
    return df
