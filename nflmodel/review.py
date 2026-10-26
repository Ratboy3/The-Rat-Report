"""Weekly review: how last week's numbers did against real DraftKings lines and results.

  python run.py weekly-review --season 2026 --week 7

Uses the last DraftKings price before each kickoff (prop_books_history.csv), the week's player stats and the
bets log. Writes review_<season>_wk<week>.md (readable summary) and appends one row per check to reviews.csv,
so patterns can be tracked across weeks before anything is changed. It does not change the model: fixes are
tested on the multi-season backtest and kept only if they do better there.

Checks
  1. Props, by stat: how often Overs hit vs DraftKings lines, how often our pricing picked the Over,
     log loss of our blended chance vs DraftKings' no-vig chance (lower = sharper), and leans' return.
  2. Anytime TDs: log loss of the model alone, the market alone, and the live 40/60 blend.
  3. Bets and leans: record, units and average CLV by market and stat.
  4. Flags: a stat whose picks lean one way 70%+ of the time, or where our chance is worse than the
     market's by 0.01+ log loss on 40+ lines, is listed as something to investigate.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import player_proj as PP
from . import props as P
from .betting import decimal, ev

CT = ZoneInfo("America/Chicago")
STAT_COL = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards",
            "receptions": "receptions", "pass_tds": "passing_tds"}


def _ll(p, y) -> float:
    p = np.clip(np.asarray(p, float), 0.02, 0.98)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _pregame_lines(history: str, games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    h = pd.read_csv(history)
    h = h[h.book == "draftkings"].copy()
    h["t"] = pd.to_datetime(h.fetched_at)
    h["ko"] = pd.to_datetime(h.kickoff, utc=True, errors="coerce").dt.tz_convert("America/Chicago").dt.tz_localize(None)
    h = h[h.t < h.ko]
    wk = games[(games.season == season) & (games.week == week)]
    keys = set(wk.away_team + " @ " + wk.home_team)
    h = h[h.game.isin(keys)]
    return h.sort_values("t").groupby(["game", "player", "stat"]).tail(1)


def run(season: int, week: int, games: pd.DataFrame, props_csv: str, history="prop_books_history.csv",
        bets_log="bets_log.csv", out_dir=".") -> dict:
    lines = _pregame_lines(history, games, season, week)
    stats = PP._load("stats", season, True)
    stats = stats[stats.week == week]
    proj = pd.read_csv(props_csv)[["player", "stat", "model_prob"]].rename(columns={"model_prob": "model"})
    by_name = {n: g.iloc[0] for n, g in stats.groupby("player_display_name")}
    out = {"season": season, "week": week, "props": [], "td": None, "log": [], "flags": []}

    # 1. yardage / count props
    rows = []
    for r in lines[lines.stat.isin(STAT_COL)].merge(proj, on=["player", "stat"], how="inner").itertuples():
        s = by_name.get(r.player)
        if s is None or pd.isna(r.line) or pd.isna(r.over_odds) or pd.isna(r.under_odds) or pd.isna(r.model):
            continue
        act = float(np.nan_to_num(s[STAT_COL[r.stat]]))
        if act == r.line:
            continue
        nv = P._no_vig_over(r.over_odds, r.under_odds)
        mm = P._market_mean(r.stat, r.line, nv)
        w = P._trust(P.PROP_MODEL_WEIGHT, r.model, mm)
        d = P._dist(r.stat, w * r.model + (1 - w) * mm)
        o = P._probs(d, r.line)
        eo, eu = ev(o, r.over_odds), ev((o[2], o[1], o[0]), r.under_odds)
        side, e = ("Over", eo) if eo >= eu else ("Under", eu)
        y = int(act > r.line)
        odds = r.over_odds if side == "Over" else r.under_odds
        pnl = (decimal(odds) - 1) if (y == 1) == (side == "Over") else -1.0
        rows.append(dict(stat=r.stat, y=y, p=o[0] / max(1e-9, 1 - o[1]), nv=nv, side=side, e=e, pnl=pnl))
    d = pd.DataFrame(rows)
    for st, g in d.groupby("stat") if len(d) else []:
        lean = g[g.e > 0]
        rec = dict(stat=st, lines=len(g), over_hit=round(g.y.mean(), 3), picked_over=round((g.side == "Over").mean(), 3),
                   ll_ours=round(_ll(g.p, g.y), 4), ll_market=round(_ll(g.nv, g.y), 4),
                   leans=len(lean), lean_roi=round(float(lean.pnl.mean()), 3) if len(lean) else np.nan)
        out["props"].append(rec)
        if len(g) >= 20 and abs(rec["picked_over"] - 0.5) >= 0.2:
            out["flags"].append(f"{st}: picks went {rec['picked_over']:.0%} Over on {len(g)} lines (Overs hit {rec['over_hit']:.0%})")
        if len(g) >= 40 and rec["ll_ours"] - rec["ll_market"] >= 0.01:
            out["flags"].append(f"{st}: our chance was less accurate than DraftKings' (log loss {rec['ll_ours']} vs {rec['ll_market']})")

    # 2. anytime TDs
    p = pd.read_csv(props_csv)
    t = lines[lines.stat == "anytime_td"].merge(p[p.stat == "anytime_td"][["player", "model_prob", "adj_projection"]],
                                               on="player", how="inner")
    tr = []
    for r in t.itertuples():
        s = by_name.get(r.player)
        if s is None or pd.isna(r.over_odds):
            continue
        y = int(float(np.nan_to_num(s.get("rushing_tds", 0))) + float(np.nan_to_num(s.get("receiving_tds", 0))) > 0)
        mk = (1 / decimal(r.over_odds)) * P.TD_BOOK_MARGIN
        tr.append(dict(y=y, model=r.model_prob, market=mk, blend=r.adj_projection))
    if tr:
        tt = pd.DataFrame(tr).dropna()
        out["td"] = dict(lines=len(tt), scored=round(tt.y.mean(), 3), ll_model=round(_ll(tt.model, tt.y), 4),
                         ll_market=round(_ll(tt.market, tt.y), 4), ll_blend=round(_ll(tt.blend, tt.y), 4))

    # 3. bets and leans
    if Path(bets_log).exists():
        l = pd.read_csv(bets_log)
        l = l[(l.season == season) & (l.week == week) & l.result.isin(["win", "loss", "push"])]
        l = l.assign(kind=np.where(l.market == "prop", "prop: " + l.stat.astype(str), l.market))
        for (tier, kind), g in l.groupby(["tier", "kind"]):
            out["log"].append(dict(tier=tier, kind=kind, n=len(g), wins=int((g.result == "win").sum()),
                                   losses=int((g.result == "loss").sum()), units=round(float(g.pnl.astype(float).sum()), 2),
                                   clv=round(float(g.clv.mean()) * 100, 2) if g.clv.notna().any() else np.nan))
    _write(out, out_dir)
    return out


def _write(out: dict, out_dir: str) -> None:
    now = datetime.now(CT).strftime("%a %b %-d, %-I:%M %p CT")
    md = [f"# Weekly review: {out['season']} week {out['week']} (run {now})", "", "## Props vs DraftKings lines"]
    if out["props"]:
        md.append(pd.DataFrame(out["props"]).to_markdown(index=False))
    if out["td"]:
        md += ["", "## Anytime TDs", pd.DataFrame([out["td"]]).to_markdown(index=False)]
    if out["log"]:
        md += ["", "## Bets and leans", pd.DataFrame(out["log"]).to_markdown(index=False)]
    md += ["", "## Flags to investigate"] + ([f"- {f}" for f in out["flags"]] or ["- none"])
    Path(out_dir, f"review_{out['season']}_wk{out['week']}.md").write_text("\n".join(md) + "\n")
    rows = [dict(season=out["season"], week=out["week"], check="prop", **r) for r in out["props"]]
    if out["td"]:
        rows.append(dict(season=out["season"], week=out["week"], check="td", **out["td"]))
    f = Path(out_dir, "reviews.csv")
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if len(old):
        old = old[~((old.season == out["season"]) & (old.week == out["week"]))]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(f, index=False)
