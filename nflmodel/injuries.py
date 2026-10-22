"""Injury impact for every position (QBs are handled separately by the backup-QB flag).

For each team: every player on this week's injury report or off the active roster is valued by
  position value x his usual snap share x chance he misses the game x how "new" the absence is.

"New" matters because team ratings already reflect players who have been out for weeks: a starter
who played every one of the team's last 4 games and is now Out counts in full; one who has missed
all 4 counts zero. The reverse also counts: a regular returning after missing recent games adds
back half his value (rust, snap limits).

Offensive losses lower the team's own projected points; defensive losses raise the opponent's.
Position values are points per game for a full-time starter replaced by a typical backup, taken
from published injury-value studies. They are estimates; the closing-line tracker is how we check them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SNAPS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"

POS_GROUP = {
    "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "T", "OT": "T", "G": "G", "OG": "G", "C": "C", "OL": "G",
    "DE": "DE", "EDGE": "DE", "OLB": "DE", "DT": "DT", "NT": "DT", "DL": "DT",
    "LB": "LB", "ILB": "LB", "MLB": "LB", "CB": "CB", "S": "S", "FS": "S", "SS": "S", "DB": "S",
}
OFFENSE = {"RB", "WR", "TE", "T", "G", "C"}
# points per game, full-time starter -> typical backup
POS_VALUE = {"RB": 0.25, "WR": 0.45, "TE": 0.30, "T": 0.45, "G": 0.25, "C": 0.30,
             "DE": 0.40, "DT": 0.30, "LB": 0.25, "CB": 0.45, "S": 0.30}
P_MISS = {"Out": 1.0, "Doubtful": 0.9, "Questionable": 0.25}
MIN_ROLE = 0.35          # only players who usually play at least 35% of snaps count
RETURN_FACTOR = 0.5      # a returning regular adds back half his value
UNIT_CAP = 4.0           # most a team's offense or defense can lose (points)
LOOKBACK = 4             # team games used to judge whether an absence is new


def _load_snaps(season: int, refresh: bool) -> pd.DataFrame | None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"snaps_{season}.csv"
    if refresh or not path.exists():
        try:
            df = pd.read_csv(SNAPS_URL.format(season=season))
            df.to_csv(path, index=False)
            return df
        except Exception as e:
            print(f"  could not download snap counts {season} ({e})")
            if not path.exists():
                return None
    return pd.read_csv(path)


def team_impacts(season: int, week: int, teams: list[str], refresh: bool = False) -> dict:
    """{team: {"off": pts lost, "def": pts lost (positive = worse), "players": [(name, pos, pts, note)]}}"""
    out = {t: {"off": 0.0, "def": 0.0, "players": []} for t in teams}
    snaps = _load_snaps(season, refresh)
    roster_p, inj_p = DATA / f"roster_{season}.csv", DATA / f"injuries_{season}.csv"
    if snaps is None or not roster_p.exists():
        return out
    roster = pd.read_csv(roster_p, low_memory=False)
    inj = pd.read_csv(inj_p, low_memory=False) if inj_p.exists() else pd.DataFrame()
    rw = roster[roster.week == min(week, roster.week.max())]
    pfr_to = rw.dropna(subset=["pfr_id"]).set_index("pfr_id")
    status_by_gsis = dict(zip(rw.gsis_id, rw.status))
    team_by_gsis = dict(zip(rw.gsis_id, rw.team))
    injw = inj[(inj.season == season) & (inj.week == week)] if len(inj) else inj
    rep = {r.gsis_id: r for r in injw.itertuples()} if len(injw) else {}

    snaps = snaps[(snaps.game_type == "REG") & (snaps.week < week)].copy()
    snaps["grp"] = snaps.position.map(POS_GROUP)
    snaps = snaps[snaps.grp.notna()]
    snaps["pct"] = np.where(snaps.grp.isin(OFFENSE), snaps.offense_pct, snaps.defense_pct)

    for team in teams:
        ts = snaps[snaps.team == team]
        if ts.empty:
            continue
        team_weeks = sorted(ts.week.unique())
        recent = team_weeks[-LOOKBACK:]
        for pid, ps in ts.groupby("pfr_player_id"):
            if pid not in pfr_to.index:
                continue
            ro = pfr_to.loc[pid]
            ro = ro.iloc[0] if isinstance(ro, pd.DataFrame) else ro
            gsis = ro.gsis_id
            if team_by_gsis.get(gsis) != team:
                continue                      # traded or released
            grp = ps.grp.iloc[-1]
            played = ps[ps.pct > 0]
            if played.empty:
                continue
            role = float(played.pct.tail(LOOKBACK).mean())   # usual share when healthy
            if role < MIN_ROLE:
                continue
            in_recent = played.week.isin(recent).sum()
            played_frac = in_recent / max(1, len(recent))
            # chance he misses this week
            r = rep.get(gsis)
            p_miss, note = 0.0, ""
            if status_by_gsis.get(gsis) not in (None, "ACT"):
                p_miss, note = 1.0, "Inactive / IR"
            elif r is not None and isinstance(r.report_status, str) and r.report_status in P_MISS:
                p_miss, note = P_MISS[r.report_status], r.report_status
            elif r is not None and isinstance(r.practice_status, str):
                if "Did Not" in r.practice_status:
                    p_miss, note = 0.30, "Missed practice"
                elif "Limited" in r.practice_status:
                    p_miss, note = 0.08, "Limited practice"
            val = POS_VALUE[grp] * role
            name = ro.full_name if isinstance(ro.full_name, str) else str(pid)
            side = "off" if grp in OFFENSE else "def"
            if p_miss > 0 and played_frac > 0:
                pts = val * p_miss * played_frac
                out[team][side] += pts
                out[team]["players"].append((name, grp, -pts, note))
            elif p_miss == 0 and status_by_gsis.get(gsis) == "ACT" and played_frac < 1 and len(played) >= 2:
                # a regular back after missing some of the recent games
                missed = 1 - played_frac
                pts = val * missed * RETURN_FACTOR
                if pts >= 0.05:
                    out[team][side] -= pts
                    out[team]["players"].append((name, grp, pts, "Returning"))
        for side in ("off", "def"):
            out[team][side] = float(np.clip(out[team][side], -UNIT_CAP, UNIT_CAP))
        out[team]["players"].sort(key=lambda x: -abs(x[2]))
    return out


def attach(df: pd.DataFrame, season: int, week: int, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Add home/away injury columns (points) to a week's games."""
    teams = sorted(set(df.home_team) | set(df.away_team))
    imp = team_impacts(season, week, teams, refresh)
    d = df.copy()
    for side in ("home", "away"):
        d[f"{side}_inj_off"] = [imp.get(t, {}).get("off", 0.0) for t in d[f"{side}_team"]]
        d[f"{side}_inj_def"] = [imp.get(t, {}).get("def", 0.0) for t in d[f"{side}_team"]]
    return d, imp
