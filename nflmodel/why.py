"""'Why this bet': two or three plain-English reasons behind each flagged bet, built from the same numbers the
model used (Oct 28, 2026). Nothing here changes a projection; it only explains one.

Game bets: the model's own line vs the market's, split into the pieces the game model adds up
(team strength, home field, rest, travel, QB, injuries, weather). Props: the player's recent production and
each adjustment the player model applied (opponent, pace/pass mix, team total, backup QB, next man up).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .betting import implied
from .model import _design, margin_features, total_features

STAT_WORD = {"pass_yds": "passing yards", "pass_tds": "passing TDs", "rush_yds": "rushing yards",
             "rec_yds": "receiving yards", "receptions": "receptions", "anytime_td": "touchdown"}
POS_WORD = {"QB": "QBs", "RB": "RBs", "WR": "WRs", "TE": "TEs"}


def _pts(x: float) -> str:
    return f"{abs(x):.1f} pt" + ("" if abs(round(x, 1)) == 1.0 else "s")


def _odds(o) -> str:
    return f"{int(o):+d}" if pd.notna(o) else "—"


def _line(team: str, margin: float) -> str:
    """Home-margin style number as a spread for `team` (positive margin = team favored)."""
    if abs(margin) < 0.05:
        return "pick'em"
    return f"{team} by {abs(margin):.1f}"


def _margin_parts(g: pd.Series, sm) -> dict:
    """Contributions (home-team points) to the pure model margin, grouped for people."""
    d = _design(pd.DataFrame([g]))
    if "home_qb_pts" in d:
        dm = d.copy()
        for side in ("home", "away"):
            dm.loc[d[f"{side}_qb_pts"].notna(), f"{side}_qb_change"] = 0
    else:
        dm = d
    feats = margin_features()
    x = dm[feats].iloc[0].astype(float).fillna(0.0).to_dict()
    b = dict(zip(["home"] + feats, sm.b_margin))
    c = {f: b[f] * x[f] for f in feats}
    inj = (-(g.get("home_inj_off", 0) or 0) - (g.get("home_inj_def", 0) or 0)
           + (g.get("away_inj_off", 0) or 0) + (g.get("away_inj_def", 0) or 0))
    qb = 0.0
    if pd.notna(g.get("home_qb_pts")) or pd.notna(g.get("away_qb_pts")):
        qb = float((g.get("home_qb_pts") or 0) - (g.get("away_qb_pts") or 0))
    qb += c.get("home_qb_change", 0) + c.get("away_qb_change", 0)
    return {
        "strength": c.get("r_margin", 0) + c.get("div_x_margin", 0) + c.get("epa_margin", 0) + c.get("sr_margin", 0),
        "home": 0.0 if g.get("neutral") else b["home"],
        "rest": c.get("home_bye", 0) + c.get("away_bye", 0) + c.get("rest_diff", 0),
        "travel": c.get("away_west_early", 0) + c.get("long_trip", 0),
        "qb": qb,
        "injuries": float(np.nan_to_num(inj)),
        "coach": c.get("agg_diff", 0),
        "manual": float((g.get("home_adj", 0) or 0) - (g.get("away_adj", 0) or 0)),
    }


def _total_parts(g: pd.Series, sm) -> dict:
    d = _design(pd.DataFrame([g]))
    feats = total_features()
    x = d[feats].iloc[0].astype(float).fillna(0.0).to_dict()
    b = dict(zip(["const"] + feats, sm.b_total))
    c = {f: b[f] * x[f] for f in feats}
    inj = (-(g.get("home_inj_off", 0) or 0) - (g.get("away_inj_off", 0) or 0)
           + (g.get("home_inj_def", 0) or 0) + (g.get("away_inj_def", 0) or 0))
    return {
        "weather": c.get("wind_excess", 0) + c.get("cold", 0) + (-2.0 * float(g.get("precip", 0) or 0)),
        "dome": c.get("dome", 0),
        "qb": c.get("qb_changes", 0),
        "injuries": float(np.nan_to_num(inj)),
        "coach": c.get("agg_sum", 0),
    }


def game_why(g: pd.Series, o: dict, sm) -> list[str]:
    """Reasons for one game bet (spread, total or moneyline)."""
    home, away = g.home_team, g.away_team
    out = []
    need = implied(o["odds"]) if pd.notna(o.get("odds")) else None
    if o["market"] in ("spread", "moneyline"):
        bet_team = home if o["side"] == "home" else away
        sign = 1 if o["side"] == "home" else -1               # converts home-margin numbers to the bet team's view
        mkt, mod, prj = g.spread_line, g.model_margin, g.proj_margin
        fav = lambda m: _line(home if m >= 0 else away, m)
        out.append(f"Our numbers make it {fav(mod)} on their own; the market has {fav(mkt)}. "
                   f"Blended with the market (we lean mostly on it), the projection is {fav(prj)}.")
        parts = _margin_parts(g, sm)
        words = {
            "strength": lambda v: f"team strength (season ratings and play-by-play efficiency) is worth {_pts(v)} to {home if v > 0 else away}",
            "home": lambda v: f"home field {_pts(v)}",
            "rest": lambda v: f"rest and byes {_pts(v)} toward {home if v > 0 else away}"
                              + (f" ({home} {int(g.home_rest)} days, {away} {int(g.away_rest)})" if pd.notna(g.get("home_rest")) else ""),
            "travel": lambda v: f"travel / body clock costs {away} {_pts(v)}",
            "qb": lambda v: f"quarterback situation {_pts(v)} toward {home if v > 0 else away}",
            "injuries": lambda v: f"injuries {_pts(v)} toward {home if v > 0 else away}",
            "coach": lambda v: f"4th-down aggressiveness {_pts(v)} toward {home if v > 0 else away}",
            "manual": lambda v: f"news adjustment {_pts(v)} toward {home if v > 0 else away}",
        }
        # the biggest pieces of the model's number, then the ones that help this bet
        ranked = sorted(((k, v) for k, v in parts.items() if abs(v) >= 0.3), key=lambda kv: -abs(kv[1]))
        helps = [(k, v) for k, v in ranked if v * sign > 0]
        lead = [ranked[0]] if ranked else []
        extra = [kv for kv in helps if kv not in lead][:2]
        bits = [words[k](v) for k, v in lead + extra]
        if bits:
            out.append("Biggest pieces: " + "; ".join(bits) + ".")
        if o["market"] == "moneyline" and need is not None:
            out.append(f"At {_odds(o['odds'])}, {bet_team} needs to win {need:.1%} of the time to break even; "
                       f"we have {o['win_prob']:.1%}.")
            if o["odds"] >= 200:
                out.append("Long-odds underdog: a small difference in win chance makes a big edge here, "
                           "so this is one of the moneylines the Tuesday review is watching.")
        elif need is not None:
            line = -g.spread_line if o["side"] == "home" else g.spread_line
            out.append(f"{bet_team} {line:+g} covers {o['win_prob']:.1%} of the time in our numbers; "
                       f"{_odds(o['odds'])} needs {need:.1%}.")
    else:  # total
        side = "Over" if o["side"] == "over" else "Under"
        out.append(f"Our numbers project {g.model_total:.1f} points on their own; the market total is {g.total_line:g}. "
                   f"Blended, the projection is {g.proj_total:.1f}.")
        parts = _total_parts(g, sm)
        bits = []
        if abs(parts["weather"]) >= 0.3 and isinstance(g.get("wx"), str):
            bits.append(f"weather ({g.wx}) {parts['weather']:+.1f}")
        for k, lab in (("injuries", "injuries"), ("qb", "QB change"), ("coach", "4th-down aggressiveness")):
            if abs(parts[k]) >= 0.3:
                bits.append(f"{lab} {parts[k]:+.1f}")
        if pd.notna(g.get("r_total")):
            bits.insert(0, f"the two offenses and defenses rate for {g.r_total:.1f} points before game factors")
        if bits:
            out.append("Biggest pieces: " + "; ".join(bits[:3]) + ".")
        if need is not None:
            out.append(f"The {side} hits {o['win_prob']:.1%} of the time in our numbers; {_odds(o['odds'])} needs {need:.1%}.")
    return out[:4]


def prop_why(r) -> list[str]:
    """Reasons for one flagged prop (a row of the props table from props.price_auto)."""
    g = lambda k, d=np.nan: getattr(r, k, d)
    stat, side = r.stat, r.best_side
    book = "FanDuel" if str(g("book", "")) == "FanDuel" else "DraftKings"
    price = r.over_odds if side == "Over" else r.under_odds
    win = r.p_over if side == "Over" else 1 - r.p_over
    out = []
    if stat == "anytime_td":
        share, ttd = g("why_share"), g("why_team_tds")
        if pd.notna(share) and pd.notna(ttd):
            out.append(f"{r.team} projects {ttd:.1f} touchdowns; {r.player} gets about {share:.0%} of the team's "
                       f"rushing/receiving TDs (red-zone touches weighted), so the model gives him {g('model_prob'):.0%} to score.")
        if pd.notna(g("why_td_games")) and g("why_td_games") > 0:
            out.append(f"He has scored in {int(g('why_td_scored'))} of {int(g('why_td_games'))} games this season.")
        mp = g("market_prob")
        if pd.notna(mp):
            out.append(f"The market (after the book's margin) says {mp:.0%}; we blend 40% model / 60% market for "
                       f"{r.adj_projection:.1%}. At {book} {_odds(price)} the break-even is {implied(price):.1%}.")
        if pd.notna(price) and price >= 600:
            out.append("Long-odds touchdown: small chance differences make big edges, so the Tuesday review is "
                       "watching these. Only same-day API prices can make it a bet.")
        return out[:4]
    word = STAT_WORD.get(stat, stat)
    l3, avg = g("why_l3"), g("why_avg")
    if pd.notna(avg):
        s = f"{r.player} averages {avg:.1f} {word} (recent games weighted)"
        s += f", {l3:.1f} over his last 3." if pd.notna(l3) else "."
        out.append(s)
    adj = []
    fo = g("why_f_opp")
    if pd.notna(fo) and abs(fo - 1) >= 0.03:
        adj.append(f"{r.opp} {'allows' if fo > 1 else 'holds'} {POS_WORD.get(r.position, r.position)} to "
                   f"{abs(fo - 1):.0%} {'more' if fo > 1 else 'fewer'} {word} than average")
    fv = g("why_f_vol")
    if pd.notna(fv) and abs(fv - 1) >= 0.03:
        adj.append(f"expected plays and pass/run mix {fv - 1:+.0%}")
    sf = g("script_factor")
    if pd.notna(sf) and abs(sf - 1) >= 0.03:
        adj.append(f"{r.team}'s team total vs its scoring average {sf - 1:+.0%}")
    fb = g("why_f_bqb")
    if pd.notna(fb) and abs(fb - 1) >= 0.01:
        adj.append(f"backup QB starting {fb - 1:+.0%}")
    fn = g("why_f_next")
    if pd.notna(fn) and abs(fn - 1) >= 0.01:
        adj.append(f"top teammate at his position out (next man up) {fn - 1:+.0%}")
    if str(g("injury", "") or ""):
        adj.append(f"injury report: {g('injury')}")
    if adj:
        out.append("Adjustments: " + "; ".join(adj[:3]) + ".")
    mp, mo = g("market_prob"), g("model_prob")
    if pd.notna(mp) and pd.notna(mo):
        out.append(f"Model {mo:.1f} vs the line's implied {mp:.1f}; blended (mostly market) {r.adj_projection:.1f}. "
                   f"{side} {r.line:g} hits {win:.1%}; {book} {_odds(price)} needs {implied(price):.1%}.")
    dkl = g("dk_line")
    if book == "FanDuel" and pd.notna(dkl) and abs(float(dkl) - float(r.line)) >= 0.5:
        better = (side == "Over" and r.line < dkl) or (side == "Under" and r.line > dkl)
        out.append(f"Line shopping: DraftKings has {dkl:g}, FanDuel {r.line:g}"
                   + (f", the better number for the {side}." if better else "."))
    cp = g("cons_p_over")
    if pd.notna(cp):
        cs = cp if side == "Over" else 1 - cp
        agree = "agree" if cs > implied(price) else "don't fully agree"
        out.append(f"Other books put the {side} at {cs:.0%}: they {agree}.")
    return out[:4]
