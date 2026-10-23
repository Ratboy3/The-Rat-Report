"""Head-coach aggressiveness: how often a coach goes for it on 4th down (and for two after a
touchdown) compared with what an average coach does in the same spot.

  python run.py build-aggression        # -> data/fourth_downs.csv (every 4th-down decision and PAT since 2006)

Expected go rate comes from a league-wide logistic fit on distance, field position, score, time
left, win probability and season (the league has become steadily more aggressive). A coach's
rating before any game is his recency-weighted "go rate over expected" on every prior decision,
shrunk toward average until he has a real sample. Ratings follow the head coach, not the team.

Output units: percentage points. +10 means he goes for it 10 points more often than an average
coach in the same situations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "fourth_downs.csv"
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
RAW = ROOT / "data" / "raw"
COLS = ["game_id", "season", "week", "season_type", "posteam", "home_team", "down", "ydstogo", "yardline_100",
        "play_type", "score_differential", "game_seconds_remaining", "half_seconds_remaining", "wp",
        "two_point_attempt", "qb_kneel"]

SEASON_DECAY = 0.75   # weight on a decision one season older
SHRINK_4TH = 40       # decisions of "average coach" prior on the 4th-down rating
SHRINK_2PT = 25       # touchdowns of prior on the two-point rating


# ---------------------------------------------------------------- extraction
def _extract(src) -> pd.DataFrame:
    d = pd.read_csv(src, usecols=COLS, low_memory=False)
    d = d[(d.season_type == "REG") & d.posteam.notna()]
    for c in ("posteam", "home_team"):
        d[c] = d[c].replace(C.TEAM_RENAMES)
    fourth = d[(d.down == 4) & d.play_type.isin(["pass", "run", "punt", "field_goal"]) & (d.qb_kneel != 1)].copy()
    fourth["kind"] = "4th"
    fourth["go"] = fourth.play_type.isin(["pass", "run"]).astype(int)
    pat = d[(d.play_type == "extra_point") | (d.two_point_attempt == 1)].copy()
    pat["kind"] = "pat"
    pat["go"] = (pat.two_point_attempt == 1).astype(int)
    keep = ["game_id", "season", "week", "posteam", "home_team", "kind", "go", "ydstogo", "yardline_100",
            "score_differential", "game_seconds_remaining", "wp"]
    out = pd.concat([fourth[keep], pat[keep]], ignore_index=True)
    return _compact(out)


def _compact(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in ("ydstogo", "yardline_100", "score_differential", "game_seconds_remaining"):
        d[c] = d[c].round().astype("Int64")
    d["wp"] = d.wp.round(3)
    d["kind"] = d.kind.map({"4th": "4", "pat": "p"}).fillna(d.kind)
    return d


def build(first: int = 2006, last: int | None = None, only: list[int] | None = None, use_local: bool = False) -> pd.DataFrame:
    """Stream each season's play-by-play from nflverse and keep only 4th downs and PATs (~2 MB total)."""
    import datetime
    last = last or datetime.date.today().year
    seasons = only or list(range(first, last + 1))
    old = pd.read_csv(FILE, dtype={"kind": str}) if FILE.exists() else pd.DataFrame()
    parts = []
    for s in seasons:
        local = RAW / f"pbp_{s}.csv.gz"
        src = local if (local.exists() and (s != last or use_local)) else URL.format(season=s)
        try:
            parts.append(_extract(src))
            print(f"  {s}: done")
        except Exception as e:
            print(f"  {s}: skipped ({e})")
    if not parts:
        return old
    new = pd.concat(parts, ignore_index=True)
    if len(old):
        new = pd.concat([old[~old.season.isin(new.season.unique())], new], ignore_index=True)
    new = new.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    FILE.parent.mkdir(parents=True, exist_ok=True)
    new = _compact(new.drop(columns=[c for c in ("half_seconds_remaining",) if c in new]))
    new.to_csv(FILE, index=False)
    print(f"Wrote {FILE.name} ({(new.kind == '4th').sum():,} 4th downs, {(new.kind == 'pat').sum():,} PATs)")
    return new


def refresh(season: int) -> None:
    """Rebuild one season, from the play-by-play file build-extras just downloaded when there is one."""
    build(only=[season], last=season, use_local=True)


# ---------------------------------------------------------------- expected go rate
def _logit(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], ridge)
    pen[0] = 1e-6
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        g = X.T @ (y - p) - pen * beta
        H = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def _features(f: pd.DataFrame, seasons: list[int]) -> np.ndarray:
    ytg = f.ydstogo.clip(1, 15).to_numpy(float)
    yl = f.yardline_100.clip(1, 99).to_numpy(float)
    sd = f.score_differential.fillna(0).clip(-21, 21).to_numpy(float)
    t = f.game_seconds_remaining.fillna(1800).to_numpy(float) / 3600
    wp = f.wp.fillna(0.5).clip(0.01, 0.99).to_numpy(float)
    cols = [np.ones(len(f)), np.log(ytg), (ytg == 1).astype(float), ytg / 10]
    for lo, hi in ((1, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 99)):
        cols.append(((yl >= lo) & (yl < hi)).astype(float))
    zone = np.where(yl < 35, 0, np.where(yl < 55, 1, 2))
    for z in (0, 1, 2):
        cols.append((zone == z) * np.log(ytg))
    cols += [sd / 10, (sd < 0) * (1 - t), (sd < -8) * (1 - t), (sd > 0) * (1 - t), t, np.log(wp / (1 - wp)) / 3]
    for s in seasons[1:]:
        cols.append((f.season.to_numpy() == s).astype(float))
    return np.column_stack(cols)


def expected(df: pd.DataFrame) -> pd.DataFrame:
    """Add xgo (league-average chance of going for it / going for two in that spot)."""
    df = df.copy()
    df["xgo"] = np.nan
    f4 = df[(df.kind == "4th") & df.wp.between(0.05, 0.95) & (df.game_seconds_remaining > 120)
            & df.ydstogo.notna() & df.yardline_100.notna()]
    seasons = sorted(f4.season.unique())
    X = _features(f4, seasons)
    b = _logit(X, f4.go.to_numpy(float))
    df.loc[f4.index, "xgo"] = 1 / (1 + np.exp(-(X @ b)))
    # two-point tries: early in games (before the 4th quarter) most coaches kick; league rate per season
    pat = df[(df.kind == "pat") & (df.game_seconds_remaining > 900)]
    rate = pat.groupby("season").go.mean()
    df.loc[pat.index, "xgo"] = pat.season.map(rate).to_numpy()
    return df


# ---------------------------------------------------------------- coach ratings
def _with_coach(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "home_coach", "away_coach"]].drop_duplicates("game_id")
    df = df.merge(g, on="game_id", how="left")
    df["coach"] = np.where(df.posteam == df.home_team, df.home_coach, df.away_coach)
    return df.drop(columns=["home_coach", "away_coach"])


def load(games: pd.DataFrame) -> pd.DataFrame | None:
    if not FILE.exists():
        return None
    raw = pd.read_csv(FILE, dtype={"kind": str})
    raw["kind"] = raw.kind.map({"4": "4th", "p": "pat"}).fillna(raw.kind)
    df = expected(raw)
    df = _with_coach(df.dropna(subset=["xgo"]), games)
    df["resid"] = df.go - df.xgo
    return df


_CACHE: dict = {}


def load_cached(games: pd.DataFrame) -> pd.DataFrame | None:
    key = (len(games), FILE.stat().st_mtime if FILE.exists() else 0)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = load(games)
    return _CACHE[key]


def ratings_before(dec: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Every coach's rating using decisions before (season, week)."""
    past = dec[(dec.season < season) | ((dec.season == season) & (dec.week < week))]
    w = SEASON_DECAY ** (season - past.season.to_numpy())
    past = past.assign(w=w, wr=w * past.resid)
    out = []
    for kind, k in (("4th", SHRINK_4TH), ("pat", SHRINK_2PT)):
        p = past[past.kind == kind].groupby("coach").agg(sw=("w", "sum"), swr=("wr", "sum"), n=("w", "size"))
        p[f"agg_{kind}"] = 100 * p.swr / (p.sw + k)
        p[f"n_{kind}"] = p.n
        out.append(p[[f"agg_{kind}", f"n_{kind}"]])
    r = out[0].join(out[1], how="outer")
    r[["agg_4th", "agg_pat"]] = r[["agg_4th", "agg_pat"]].fillna(0.0)
    r[["n_4th", "n_pat"]] = r[["n_4th", "n_pat"]].fillna(0).astype(int)
    return r.reset_index()


def attach(df: pd.DataFrame, dec: pd.DataFrame | None) -> pd.DataFrame:
    """Add home_agg/away_agg (4th-down, pct pts over expected) and home_agg2/away_agg2 (two-point) to games."""
    df = df.copy()
    for c in ("home_agg", "away_agg", "home_agg2", "away_agg2"):
        df[c] = 0.0
    if dec is None or "home_coach" not in df:
        return df
    for (s, wk), idx in df.groupby(["season", "week"]).groups.items():
        r = ratings_before(dec, int(s), int(wk)).set_index("coach")
        for side in ("home", "away"):
            co = df.loc[idx, f"{side}_coach"]
            df.loc[idx, f"{side}_agg"] = co.map(r.agg_4th).fillna(0.0).to_numpy()
            df.loc[idx, f"{side}_agg2"] = co.map(r.agg_pat).fillna(0.0).to_numpy()
    return df
