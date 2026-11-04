"""Keep bets_log.csv and lines_history.csv safe when several writers touch them (Nov 3, 2026).

Since Nov 3 the website rebuild (GitHub Actions) logs new bets, records closing prices and grades results on
every run, so tracking no longer stops when the claude.ai runs do. The claude.ai runs still push their own copy
of both files, which can be older. Every rebuild therefore merges: the repo's file, the bot's last copy
(data/bets_log_bot.csv, data/lines_history_bot.csv, written only by the rebuild) and any extra file given.
Rows are matched on bet_id (bets) or time + game (lines); for a bet seen twice the most complete row wins:
graded beats ungraded, then the later closing-price check.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BOT = {"bets_log.csv": ROOT / "data" / "bets_log_bot.csv", "lines_history.csv": ROOT / "data" / "lines_history_bot.csv"}


def _ts(s: pd.Series, season: pd.Series | None = None) -> pd.Series:
    """'Sun Nov 1, 10:04 AM CT' -> timestamp (year from the season, or the current year)."""
    # year: the one closest to the middle of the season (Sep-Feb), so January stamps land in the right year
    # (Nov 4, 2026 audit: a Dec 31 close was beating a Jan 2 one)
    from .props import _parse_asof
    if season is None:
        ref = pd.Series(pd.Timestamp.now(), index=s.index)
    else:
        ref = pd.to_datetime(season.astype("Int64").astype(str) + "-11-15", errors="coerce")
    return pd.Series([_parse_asof(a, r) if pd.notna(r) else pd.NaT for a, r in zip(s, ref)], index=s.index,
                     dtype="datetime64[ns]")


def merge_bets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    cols = list(dict.fromkeys(c for f in frames for c in f.columns))
    d = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)
    d["_order"] = range(len(d))
    first = d.groupby("bet_id")["_order"].min()
    d["_graded"] = d["result"].isin(["win", "loss", "push", "void"]).astype(int)
    d["_close"] = _ts(d["close_at"].fillna(""), d.get("season")).fillna(pd.Timestamp.min)
    d = d.sort_values(["bet_id", "_graded", "_close"]).groupby("bet_id", as_index=False).tail(1)
    d["_first"] = d["bet_id"].map(first)
    return d.sort_values("_first").drop(columns=["_order", "_graded", "_close", "_first"]).reset_index(drop=True)


def merge_lines(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["at", "game_id"], keep="last")
    d["_t"] = _ts(d["at"])
    return d.sort_values("_t", kind="stable").drop(columns="_t").reset_index(drop=True)


def merge_file(name: str, extra: list[str] | None = None, save_bot: bool = False) -> int:
    """Merge ROOT/name with the bot copy and any extra files; write ROOT/name (and the bot copy if asked)."""
    paths = [ROOT / name, BOT[name]] + [Path(x) for x in (extra or [])]
    frames = [pd.read_csv(p) for p in paths if p.exists() and p.stat().st_size > 0]
    out = merge_bets(frames) if name == "bets_log.csv" else merge_lines(frames)
    if out.empty:
        return 0
    out.to_csv(ROOT / name, index=False)
    if save_bot:
        BOT[name].parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(BOT[name], index=False)
    return len(out)


def merge_all(extra_dir: str | None = None, save_bot: bool = False) -> None:
    for name in BOT:
        extra = [str(Path(extra_dir) / name)] if extra_dir else None
        n = merge_file(name, extra, save_bot)
        print(f"  {name}: {n} rows after merge")
