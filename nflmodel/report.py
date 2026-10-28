"""Render the weekly model output as one self-contained, readable HTML page."""
from __future__ import annotations

from datetime import datetime, timedelta
from html import escape

import numpy as np
import pandas as pd

from . import config as C
from .betting import decimal, fair_american, implied
from . import why as W

NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
    "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers",
    "LV": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}
MARKET_LABEL = {"spread": "Spread", "total": "Total", "moneyline": "Moneyline"}
STAT_LABEL = {"pass_yds": "Passing yards", "rush_yds": "Rushing yards", "rec_yds": "Receiving yards",
              "receptions": "Receptions", "pass_tds": "Passing TDs", "anytime_td": "Anytime TD"}


def _odds(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{int(round(x)):+d}"


def _spread(team: str, x: float) -> str:
    x = round(x * 2) / 2
    return f"{team} {'PK' if x == 0 else f'{x:+.1f}'}"


def _kickoff(g) -> str:
    try:
        et = datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M")
        ct = et - timedelta(hours=1)
        return ct.strftime("%a %b %-d · %-I:%M %p CT")
    except (TypeError, ValueError):
        return str(g.gameday)


def _pct(x, digits=0) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def _bk(r) -> str:
    """Short label for the book a price comes from: DK (DraftKings) unless the bet is priced at FanDuel."""
    return "FD" if str(getattr(r, "book", "")) == "FanDuel" else "DK"


def _signed_pct(x, digits=1) -> str:
    return "—" if pd.isna(x) else f"{x * 100:+.{digits}f}%"


# ------------------------------------------------------------------ sections
def _bets_section(games) -> str:
    rows = []
    for item in games:
        g = item["g"]
        for o in item["opts"]:
            if o["ev"] >= C.MIN_EV:
                rows.append((o["ev"], g, o))
    rows.sort(key=lambda r: -r[0])
    if not rows:
        return ('<p class="empty">No game bet clears the ' f'{C.MIN_EV:.0%}' ' bar right now.</p>')
    body = "".join(
        f"""<tr><td><b>{escape(o['pick'])}</b><span class="cell-sub">{g.away_team} @ {g.home_team} · {_kickoff(g)}</span></td>
        <td class="n">{_odds(o['odds'])}</td>
        <td class="n" title="Break-even {_pct(implied(o['odds']), 1)}">{_pct(o['win_prob'], 1)}</td>
        <td class="n good">{_signed_pct(ev)}</td></tr>"""
        for ev, g, o in rows)
    return f"""<div class="tablebox"><table>
      <thead><tr><th>Bet</th><th class="n">Odds</th><th class="n">Win %</th><th class="n">Edge</th></tr></thead>
      <tbody>{body}</tbody></table></div>
"""


def _books_note(r) -> str:
    """'Other books: 54%' for the side we like, from the consensus of the other sportsbooks at DraftKings' line."""
    cp = getattr(r, "cons_p_over", np.nan)
    if pd.isna(cp) or not getattr(r, "n_books", 0):
        return ""
    side = cp if r.best_side == "Over" else 1 - cp
    ce = getattr(r, "cons_ev", np.nan)
    tone = "" if pd.isna(ce) else (" good" if ce > 0 else " bad")
    return f' · <span class="books{tone}" title="Consensus of {int(r.n_books)} other sportsbooks at this line">other books {side:.0%}</span>'


def _top_props(props: pd.DataFrame | None, n: int = 5) -> str:
    if props is None or props.empty or "line" not in props:
        return '<p class="empty">No prop lines yet.</p>'
    have = props[props.line.notna() & (props.ev > 0)].sort_values("ev", ascending=False).head(n)
    if have.empty:
        found = int(props.line.notna().sum())
        return (f'<p class="empty">No positive prop edges yet ({found} DraftKings lines found). '
                'Lines fill in through the weekend.</p>')
    rows = "".join(
        f"""<tr><td><b>{escape(r.player)}</b> {'Anytime TD' if r.stat == 'anytime_td' else f'{r.best_side} {r.line:g} {STAT_LABEL.get(r.stat, r.stat).lower()}'}
        <span class="cell-sub">{r.team} vs {r.opp} · {_bk(r)} {escape(r.as_of)}{_books_note(r)}</span>{'<span class="inj">Large edge: check news and the current line</span>' if r.ev > 0.15 else ''}</td>
        <td class="n">{_odds(r.over_odds if r.best_side == 'Over' else r.under_odds)}</td>
        <td class="n">{_pct(r.p_over if r.best_side == 'Over' else 1 - r.p_over, 1)}</td>
        <td class="n {'good' if r.flag else ''}">{_signed_pct(r.ev)}{'' if r.flag else ' <span class="muted">lean</span>'}</td></tr>"""
        for r in have.itertuples())
    return f"""<div class="tablebox"><table>
      <thead><tr><th>Prop</th><th class="n">Odds</th><th class="n">Win %</th><th class="n">Edge</th></tr></thead>
      <tbody>{rows}</tbody></table></div>"""


def _leans_section(live, props, log) -> str:
    """Positive edges below the betting bar: open ones for games not yet played, then how past leans did."""
    from .clv import summary
    live_keys = {f"{it['g'].away_team} @ {it['g'].home_team}" for it in live}
    # record strip (kept apart from the bet record)
    strip = ""
    sm_ = summary(log) if log is not None and len(log) else None
    if sm_:
        cells = []
        for k, lab in (("Leans: games", "Game leans"), ("Leans: props", "Prop leans")):
            v = sm_.get(k)
            if not v or not v["bets"]:
                cells.append(f'<div><span>{lab}</span><b>—</b><em>none logged yet</em></div>')
                continue
            clv = _signed_pct(v["clv"]) if v["with_close"] else "—"
            cls = "good" if v["with_close"] and v["clv"] > 0 else ("bad" if v["with_close"] and v["clv"] < 0 else "")
            pushes = v["settled"] - v["wins"] - v["losses"]
            cells.append(f'<div><span>{lab}</span><b>{v["wins"]}-{v["losses"]}{f"-{pushes}" if pushes else ""}</b>'
                         f'<em>{v["units"]:+.1f} units · avg CLV <i class="{cls}">{clv}</i> · {v["bets"]} logged</em></div>')
        strip = f'<div class="leanstrip">{"".join(cells)}</div>'

    # open game leans
    grows = []
    for it in live:
        g = it["g"]
        for o in it["opts"]:
            if 0 < o["ev"] < C.MIN_EV:
                grows.append((o["ev"], g, o))
    grows.sort(key=lambda r: -r[0])
    games_html = ('<p class="empty">No game leans right now.</p>' if not grows else
                  f"""<div class="tablebox"><table><thead><tr><th>Lean</th><th class="n">Odds</th><th class="n">Win %</th><th class="n">Edge</th></tr></thead>
      <tbody>{''.join(f'''<tr><td><b>{escape(o['pick'])}</b><span class="cell-sub">{g.away_team} @ {g.home_team} · {_kickoff(g)}</span></td>
        <td class="n">{_odds(o['odds'])}</td><td class="n" title="Break-even {_pct(implied(o['odds']), 1)}">{_pct(o['win_prob'], 1)}</td>
        <td class="n">{_signed_pct(ev)}</td></tr>''' for ev, g, o in grows)}</tbody></table></div>""")

    # open prop leans: biggest edges shown, the rest behind a toggle
    props_html = '<p class="empty">No prop leans right now.</p>'
    if props is not None and not props.empty and "line" in props:
        pl = props[props.line.notna() & (props.ev > 0) & ~props.flag.astype(bool) & props.game.isin(live_keys)]
        pl = pl.sort_values("ev", ascending=False)
        if len(pl):
            def prow(r):
                what = "Anytime TD" if r.stat == "anytime_td" else f"{r.best_side} {r.line:g} {STAT_LABEL.get(r.stat, r.stat).lower()}"
                return (f"""<tr><td><b>{escape(r.player)}</b> {what}<span class="cell-sub">{escape(r.game)} · {_bk(r)} {escape(str(r.as_of))}{_books_note(r)}</span></td>
        <td class="n">{_odds(r.over_odds if r.best_side == 'Over' else r.under_odds)}</td>
        <td class="n">{_pct(r.p_over if r.best_side == 'Over' else 1 - r.p_over, 1)}</td><td class="n">{_signed_pct(r.ev)}</td></tr>""")
            head = '<thead><tr><th>Lean</th><th class="n">Odds</th><th class="n">Win %</th><th class="n">Edge</th></tr></thead>'
            top, rest = pl.head(10), pl.iloc[10:]
            props_html = f'<div class="tablebox"><table>{head}<tbody>{"".join(prow(r) for r in top.itertuples())}</tbody></table></div>'
            if len(rest):
                props_html += (f'<details class="more"><summary>Show all {len(pl)} prop leans</summary><div class="tablebox"><table>{head}'
                               f'<tbody>{"".join(prow(r) for r in rest.itertuples())}</tbody></table></div></details>')

    # past leans (kickoff passed): newest first, with closing line value and result
    past_html = ""
    if log is not None and len(log) and "tier" in log:
        lg = log[log.tier.fillna("bet") == "lean"].copy()
        ko = pd.to_datetime(lg.kickoff, utc=True, errors="coerce")
        lg = lg[ko <= pd.Timestamp.now(tz="UTC")].assign(_ko=ko).sort_values(["_ko", "ev"], ascending=[False, False])
        if len(lg):
            def res(r):
                if isinstance(r.result, str) and r.result:
                    c = "good" if r.result == "win" else ("bad" if r.result == "loss" else "")
                    pnl = f" ({float(r.pnl):+.2f}u)" if pd.notna(r.pnl) else ""
                    return f'<span class="{c}">{escape(r.result.capitalize())}</span><span class="cell-sub">{pnl.strip()}</span>'
                return '<span class="muted">Awaiting stats</span>' if r.market == "prop" else '<span class="muted">Pending</span>'
            rows = "".join(
                f"""<tr><td>{escape(_pretty_pick(r.pick))}<span class="cell-sub">{escape(str(r.game))} · logged {escape(str(r.logged_at))}</span></td>
        <td class="n">{_odds(r.odds)}</td><td class="n hide-sm">{_signed_pct(r.ev)}</td>
        <td class="n {'good' if pd.notna(r.clv) and r.clv > 0 else 'bad' if pd.notna(r.clv) and r.clv < 0 else ''}">{_signed_pct(r.clv) if pd.notna(r.clv) else '—'}</td>
        <td>{res(r)}</td></tr>""" for r in lg.head(40).itertuples())
            past_html = f"""<details class="more past-leans"><summary>Played leans ({len(lg)}, newest first)</summary>
    <div class="tablebox"><table><thead><tr><th>Lean</th><th class="n">Price</th><th class="n hide-sm">Edge then</th><th class="n">CLV</th><th>Result</th></tr></thead>
    <tbody>{rows}</tbody></table></div></details>"""

    return f"""<section id="leans"><h2>Leans</h2>
    <p class="lede">Positive edges below the {C.MIN_EV:.0%} betting bar. <b>These are not bets.</b> They're tracked to test whether the bar is set right:
    if leans keep beating the closing line, some of them should be bets; if they don't, the bar is doing its job.
    Their record is kept separate from the bet record.</p>
    {strip}
    <div class="best">
      <div><h3>Game leans</h3>{games_html}</div>
      <div><h3>Prop leans</h3>{props_html}</div>
    </div>
    {past_html}
    <p class="note">Leans are small edges, so a few dozen results say little. Judge them by average CLV (how the price compared with the closing line) before win-loss.
    Each lean is logged at the first price it appeared; finished games move to Played with the result.</p></section>"""


def _pretty_pick(pick) -> str:
    """'Caleb Williams Under 247.5 pass_yds' -> 'Caleb Williams Under 247.5 passing yards'."""
    t = str(pick)
    for k, v in STAT_LABEL.items():
        t = t.replace(f" {k}", f" {v.lower()}")
    return t


def _short_inj(text: str) -> str:
    """Compact injury note for tight lists: 'Limited in practice (Not injury related - resting player)' -> 'Limited (rest day)'."""
    t = str(text)
    t = t.replace("Not injury related - resting player", "rest day").replace("Not injury related", "not injury")
    t = t.replace("Limited in practice", "Limited").replace("Missed practice", "Missed practice")
    return t


def _started(g) -> bool:
    """Kicked off already (nflverse kickoff times are Eastern)."""
    try:
        from zoneinfo import ZoneInfo
        k = datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
        return k <= datetime.now(ZoneInfo("America/Chicago"))
    except (TypeError, ValueError, AttributeError):
        return False


def _final_from_news(g, note: str):
    """'Final: Steelers 16, Saints 10.' in the game's news note -> the game with scores filled in, so a finished
    game moves to the bottom right away (the nflverse feed posts Sunday scores hours later)."""
    import re
    m = re.match(r"\s*Final:\s*(.+?)\s+(\d+),\s*(.+?)\s+(\d+)\b", str(note or ""))
    if not m:
        return None
    by_name = {v.lower(): k for k, v in NAMES.items()}
    pts = {}
    for name, sc in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
        t = by_name.get(name.strip().lower()) or (name.strip().upper() if name.strip().upper() in NAMES else None)
        if t is None:
            return None
        pts[t] = int(sc)
    if set(pts) != {g.home_team, g.away_team}:
        return None
    g = g.copy()
    g["home_score"], g["away_score"] = pts[g.home_team], pts[g.away_team]
    g["result"] = pts[g.home_team] - pts[g.away_team]
    g["total"] = pts[g.home_team] + pts[g.away_team]
    return g


def _is_final(g) -> bool:
    """Game finished and scored in the nflverse feed."""
    return all(pd.notna(g.get(c)) for c in ("home_score", "away_score", "result"))


def _game_card(item, news=None, tds=None, props=None, gi=0, inj=None, mkt=None) -> str:
    g, opts, hw = item["g"], item["opts"], item["home_win"]
    flagged = [o for o in opts if o["ev"] >= C.MIN_EV]
    best = {}
    for o in opts:
        if o["market"] not in best or o["ev"] > best[o["market"]]["ev"]:
            best[o["market"]] = o
    tags = []
    if g.neutral:
        tags.append("Neutral site")
    if g.div_game:
        tags.append("Division")
    if g.dome:
        tags.append("Dome")
    if g.home_bye:
        tags.append(f"{g.home_team} off bye")
    if g.away_bye:
        tags.append(f"{g.away_team} off bye")
    mkt = mkt or {}
    qst = mkt.get("qb_status") or {}
    for t, side in ((g.away_team, "away"), (g.home_team, "home")):
        q = qst.get(t)
        if q and q["kind"] == "backup":
            tags.append(f"{t} backup QB: {q['qb']} for {q['other']}")
        elif q and q["kind"] == "returning":
            tags.append(f"{t} QB {q['qb'].split()[-1]} returns")
        elif q and q["kind"] == "unclear":
            tags.append(f"⚠ {t} QB to confirm")
        elif not q and g.get(f"{side}_qb_change_t", g.get(f"{side}_qb_change", 0)):
            tags.append(f"{t} QB change")
    if g.away_west_early:
        tags.append(f"{g.away_team} early body clock")
    if not g.dome and isinstance(g.get("wx"), str):
        tags.append(("⚠ " if g.get("wind_excess", 0) > 0 or g.get("cold", 0) or g.get("precip", 0) else "") + g.wx)
    tend = mkt.get("tendencies")
    if tend is not None:
        for t in (g.away_team, g.home_team):
            if t not in tend.index:
                continue
            r = tend.loc[t]
            if r.sec_per_play < r.sec_per_play_lg - 2:
                tags.append(f"{t} fast pace")
            elif r.sec_per_play > r.sec_per_play_lg + 2:
                tags.append(f"{t} slow pace")
            if r.proe > 0.04:
                tags.append(f"{t} pass-heavy")
            elif r.proe < -0.04:
                tags.append(f"{t} run-heavy")
    for t, side in ((g.away_team, "away"), (g.home_team, "home")):
        ag = g.get(f"{side}_agg", 0.0)
        if pd.notna(ag) and ag >= 3:
            tags.append(f"{t} aggressive on 4th")
        elif pd.notna(ag) and ag <= -3:
            tags.append(f"{t} conservative on 4th")
    game_key = f"{g.away_team} @ {g.home_team}"
    best_px = mkt.get("best_games", {})
    tag_html = "".join(f'<span class="tag">{escape(t)}</span>' for t in tags)

    def lean(market):
        o = best.get(market)
        if not o:
            return ""
        cls = "lean bet" if o["ev"] >= C.MIN_EV else ("lean" if o["ev"] > 0 else "lean pass")
        word = "Bet" if o["ev"] >= C.MIN_EV else ("Lean" if o["ev"] > 0 else "No edge ·")
        bp = best_px.get((game_key, market, o["side"]))
        alt = ""
        if bp:
            book, bl, bo = bp[0]
            line_now = g.spread_line if market == "spread" else (g.total_line if market == "total" else None)
            better = (book not in ("DraftKings", o.get("book", "DraftKings"))) and (bo != o["odds"] or (line_now is not None and pd.notna(bl)
                                                                     and abs(abs(bl) - abs(line_now)) > 0.01))
            if better:
                ln = "" if market == "moneyline" or pd.isna(bl) else (f"{bl:+g} " if market == "spread" else f"{bl:g} ")
                alt = f'<span class="cell-sub">Best: {escape(book)} {ln}{_odds(bo)}</span>'
        return f'<div class="{cls}">{word} {escape(o["pick"])} <span class="n">{_signed_pct(o["ev"])}</span>{alt}</div>'

    fav_home = g.proj_margin >= 0
    done = _is_final(g)
    if done:
        hs, as_ = int(g.home_score), int(g.away_score)
        a_pts = f'<b>{as_}</b>' if as_ > hs else f'{as_}'
        h_pts = f'<b>{hs}</b>' if hs > as_ else f'{hs}'
        when = (f'<b class="final-lbl">Final</b> · {_kickoff(g)}'
                f'<span class="cell-sub">Projected {g.away_team} {g.proj_away_pts:.1f}, {g.home_team} {g.proj_home_pts:.1f}</span>')
    else:
        a_pts, h_pts, when = f"{g.proj_away_pts:.1f}", f"{g.proj_home_pts:.1f}", _kickoff(g)
    # phone summary (shown while the card is collapsed): market lines plus the best bet or lean
    top_o = max(opts, key=lambda o: o["ev"]) if opts else None
    pick_html = ""
    if top_o is not None and top_o["ev"] > 0 and not done:
        isbet = top_o["ev"] >= C.MIN_EV
        pick_html = (f'<span class="sum-pick{" bet" if isbet else ""}">{"Bet" if isbet else "Lean"} {escape(top_o["pick"])} '
                     f'<span class="n">{_signed_pct(top_o["ev"])}</span></span>')
    sum_html = (f'<div class="card-sum n"><span>{_spread(g.home_team, -g.spread_line)}</span><span>O/U {g.total_line:.1f}</span>'
                f'<span>{g.away_team} {_odds(g.away_moneyline)} · {g.home_team} {_odds(g.home_moneyline)}</span>{pick_html}</div>'
                '<button type="button" class="card-more" aria-expanded="false">Lines, props and news</button>')
    return f"""<article class="card{' has-bet' if flagged else ''}{' final' if done else ''}">
      <header class="card-head">
        <div class="teams">
          <div class="team"><span class="abbr">{g.away_team}</span><span class="nm">{NAMES.get(g.away_team, '')}</span>
            <span class="qb">{escape(str(g.away_qb_name or ''))}</span><span class="pts n">{a_pts}</span></div>
          <div class="team"><span class="abbr">{g.home_team}</span><span class="nm">{NAMES.get(g.home_team, '')}</span>
            <span class="qb">{escape(str(g.home_qb_name or ''))}</span><span class="pts n">{h_pts}</span></div>
        </div>
        <div class="when">{when}</div>
        <div class="tags">{tag_html}</div>
        {_movement_html(g, mkt)}
      </header>
      {sum_html}
      <div class="card-body">
      <div class="lines">
        <div class="line"><h4>Spread</h4>
          <dl><dt>Market</dt><dd class="n">{_spread(g.home_team, -g.spread_line)}</dd>
          <dt>Pure model</dt><dd class="n">{_spread(g.home_team, -g.model_margin)}</dd>
          <dt>Projection</dt><dd class="n"><b>{_spread(g.home_team, -g.proj_margin)}</b></dd></dl>{lean('spread')}</div>
        <div class="line"><h4>Total</h4>
          <dl><dt>Market</dt><dd class="n">{g.total_line:.1f}</dd>
          <dt>Pure model</dt><dd class="n">{g.model_total:.1f}</dd>
          <dt>Projection</dt><dd class="n"><b>{g.proj_total:.1f}</b></dd></dl>{lean('total')}</div>
        <div class="line"><h4>Win chance</h4>
          <div class="wp"><div class="wp-bar"><i style="width:{(1 - hw) * 100:.1f}%"></i></div>
            <div class="wp-lbl n"><span>{g.away_team} {_pct(1 - hw)}</span><span>{g.home_team} {_pct(hw)}</span></div></div>
          <dl><dt>Fair price</dt><dd class="n">{g.home_team} {_odds(fair_american(hw))}</dd>
          <dt>Book</dt><dd class="n">{g.home_team} {_odds(g.home_moneyline)}</dd>
          <dt></dt><dd class="n">{g.away_team} {_odds(g.away_moneyline)}</dd></dl>
          {lean('moneyline')}</div>
      </div>
      <details class="xtra" open><summary>Touchdown scorers and props</summary><div class="card-extra">{_card_tds(tds, (mkt or {}).get('best_props', {}))}{_card_props(props, gi, (mkt or {}).get('best_props', {}))}</div></details>
      {_card_injuries(g, inj, (mkt or {}).get('qb'))}
      {_card_news(news, (mkt or {}).get('dk_used', {}).get(game_key))}
      </div>
    </article>"""


def _movement_html(g, mkt) -> str:
    from .market import movement
    mv = movement(mkt.get("history") if mkt else None, g.game_id)
    src = (mkt or {}).get("dk_used", {}).get(f"{g.away_team} @ {g.home_team}")
    bits = []
    if mv:
        (s0, s1), (t0, t1) = mv["spread"], mv["total"]
        if pd.notna(s0) and pd.notna(s1) and s0 != s1:
            fav = lambda x: _spread(g.home_team, -x)
            bits.append(f"Spread {fav(s0)} → {fav(s1)}")
        if pd.notna(t0) and pd.notna(t1) and t0 != t1:
            bits.append(f"Total {t0:g} → {t1:g}")
        if bits:
            bits[0] = f"Since {escape(str(mv['since']))}: " + bits[0]
    return f'<div class="move cell-sub">{" · ".join(bits)}</div>' if bits else ""


def _clv_section(log) -> str:
    """Live results: every flagged bet against the closing line, then win/loss once settled."""
    from .clv import summary
    sm_ = summary(log)
    if not sm_:
        return ('<h3>Closing line value (live)</h3><p class="note">Starting now: every flagged bet is logged at its price and '
                'compared with the closing DraftKings price at kickoff. Results appear after the first games close.</p>')
    rows = "".join(
        f"""<tr><td><b>{k}</b></td><td class="n">{v['bets']}</td><td class="n">{v['with_close']}</td>
        <td class="n {'good' if v['clv'] > 0 else 'bad' if v['clv'] < 0 else ''}">{_signed_pct(v['clv']) if v['with_close'] else '—'}</td>
        <td class="n">{f"{v['beat']:.0%}" if v['with_close'] else '—'}</td>
        <td class="n">{v['wins']}-{v['losses']}</td><td class="n">{v['units']:+.1f}</td></tr>"""
        for k, v in sm_.items() if v["bets"])
    tier = log["tier"].fillna("bet") if "tier" in log else pd.Series("bet", index=log.index)
    recent = log[tier == "bet"].sort_values("logged_at").tail(12).iloc[::-1]
    rec_rows = "".join(
        f"""<tr><td>{escape(str(r.pick))}<span class="cell-sub">{escape(str(r.game))} · logged {escape(str(r.logged_at))}</span></td>
        <td class="n">{_odds(r.odds)}</td><td class="n">{_signed_pct(r.ev)}</td>
        <td class="n {'good' if pd.notna(r.clv) and r.clv > 0 else 'bad' if pd.notna(r.clv) and r.clv < 0 else ''}">{_signed_pct(r.clv) if pd.notna(r.clv) else '—'}</td>
        <td>{escape(str(r.result)) if isinstance(r.result, str) and r.result else 'Open'}</td></tr>"""
        for r in recent.itertuples())
    lean_log = log[tier == "lean"]
    settled = lean_log[lean_log.result.isin(["win", "loss", "push"])].sort_values("logged_at").iloc[::-1].head(15)
    lean_rows = "".join(
        f"""<tr><td>{escape(str(r.pick))}<span class="cell-sub">{escape(str(r.game))}</span></td>
        <td class="n">{_odds(r.odds)}</td><td class="n">{_signed_pct(r.ev)}</td>
        <td>{escape(str(r.result)).capitalize()}</td></tr>""" for r in settled.itertuples())
    lean_more = (f"""<details class="more"><summary>Latest settled leans</summary><div class="tablebox"><table><thead><tr><th>Lean</th>
    <th class="n">Price</th><th class="n">Edge then</th><th>Result</th></tr></thead><tbody>{lean_rows}</tbody></table></div></details>"""
                 if lean_rows else "")
    from .clv import timing
    tt = timing(log)
    timing_html = ""
    if tt is not None and len(tt):
        trs = "".join(f"""<tr><td>{r.when}</td><td>{'Bets' if r.tier == 'bet' else 'Leans'}</td><td class="n">{r.n}</td>
        <td class="n {'good' if r.clv > 0 else 'bad' if r.clv < 0 else ''}">{_signed_pct(r.clv)}</td><td class="n">{r.beat:.0%}</td></tr>"""
                      for r in tt.itertuples())
        timing_html = f"""<details class="more"><summary>Bet early or wait? CLV by time before kickoff</summary>
    <p class="note">If bets logged early beat the close more often than ones logged late, the model's edges get priced in during the week
    and it pays to bet as soon as they show up. Too few closed bets yet to say; this fills in each week.</p>
    <div class="tablebox"><table><thead><tr><th>Logged</th><th>Type</th><th class="n">Closed</th><th class="n">Avg CLV</th><th class="n">Beat close</th></tr></thead>
    <tbody>{trs}</tbody></table></div></details>"""
    return f"""<h3>Closing line value (live)</h3>
    <p class="lede">Every flagged bet is logged at the price when it was first flagged, then re-priced at the closing
    DraftKings line. Positive CLV means the bet was worth more than the closing market said: the earliest reliable sign of a real edge.
    Leans (small positive edges below the 3% bar) are tracked the same way in their own rows, so they never mix into the bet record.</p>
    <div class="tablebox"><table><thead><tr><th>Type</th><th class="n">Logged</th><th class="n">Closed</th><th class="n">Avg CLV</th>
    <th class="n">Beat close</th><th class="n">W-L</th><th class="n">Units</th></tr></thead><tbody>{rows}</tbody></table></div>
    <details class="more"><summary>Latest logged bets</summary><div class="tablebox"><table><thead><tr><th>Bet</th><th class="n">Price</th>
    <th class="n">Edge then</th><th class="n">CLV</th><th>Result</th></tr></thead><tbody>{rec_rows}</tbody></table></div></details>
    {lean_more}
    {timing_html}"""


def _prop_bt_section(bt) -> str:
    if bt is None or len(bt) == 0:
        return ""
    lab = {"pass_yds": "Passing yards", "pass_tds": "Passing TDs", "rush_yds": "Rushing yards", "rec_yds": "Receiving yards",
           "receptions": "Receptions", "anytime_td": "Anytime TD"}
    rows = "".join(
        f"""<tr><td><b>{lab.get(r.stat, r.stat)}</b></td><td class="n">{int(r.n):,}</td>
        <td class="n">{'—' if pd.isna(r.mae) else f'{r.mae:.1f}'}</td><td class="n">{'—' if pd.isna(r.mae_base) else f'{r.mae_base:.1f}'}</td>
        <td class="n">{r.bias:+.2f}</td><td class="n">{'—' if pd.isna(r.brier) else f'{r.brier:.3f}'}</td></tr>"""
        for r in bt.itertuples())
    return f"""<h3>Prop model, past weeks</h3>
    <p class="lede">Player projections replayed for 2025 (weeks 5-18) and 2026 so far, using only information available before each game,
    then compared with what players actually did. "Simple average" is a plain recent-games average for comparison.</p>
    <div class="tablebox"><table><thead><tr><th>Stat</th><th class="n">Player-games</th><th class="n">Avg miss</th>
    <th class="n">Simple average</th><th class="n">Bias</th><th class="n">TD accuracy (Brier)</th></tr></thead><tbody>{rows}</tbody></table></div>
    <p class="note">Avg miss is the typical gap between projection and result (lower is better). Bias above zero means players beat the projections.
    Brier score: lower is better; 0.25 is a coin flip.</p>"""


def _card_injuries(g, inj, qbs=None) -> str:
    """Net injury effect on each team, with the players behind it, plus any QB change."""
    parts = []
    for t in (g.away_team, g.home_team):
        q = (qbs or {}).get(t)
        if not q:
            continue
        parts.append(f'<div><b>{t}</b> <span class="n {"bad" if q["pts"] < -0.05 else ("good" if q["pts"] > 0.05 else "")}">{q["pts"]:+.1f} pts</span>'
                     f' <span class="muted">QB {escape(str(q["new"]))} ({q["new_rating"]:+.2f} EPA/dropback) in for '
                     f'{escape(str(q["usual"]))} ({q["usual_rating"]:+.2f}), spread only</span></div>')
    if not inj and not parts:
        return ""
    inj = inj or {}
    for t in (g.away_team, g.home_team):
        v = inj.get(t)
        if not v or not v["players"]:
            continue
        net = -(v["off"] + v["def"])
        who = ", ".join(f"{escape(n.split()[-1] if len(n) > 18 else n)} {p} {x:+.1f}" + (f" ({escape(st)})" if st else "")
                        for n, p, x, st in v["players"][:4] if abs(x) >= 0.05)
        if not who:
            continue
        parts.append(f'<div><b>{t}</b> <span class="n {"bad" if net < -0.05 else ("good" if net > 0.05 else "")}">{net:+.1f} pts</span>'
                     f' <span class="muted">{who}</span></div>')
    if not parts:
        return ""
    return (f'<div class="card-inj"><h4>QB changes and injuries (points)</h4>{"".join(parts)}'
            '<p class="cell-sub">Starters out or limited, weighted by snap share and how new the absence is. '
            'Already in the pure model line.</p></div>')


TD_PER_TEAM = 3   # most likely scorers shown per team, so both sides of every game are listed


def _card_tds(tds, best=None) -> str:
    """Most likely anytime touchdown scorers in a game (top 3 per team): chance, fair odds and the DraftKings price,
    plus the best touchdown value in the game (the price that beats fair odds by the most)."""
    if tds is None or len(tds) == 0:
        return ""
    best = best or {}
    top = (tds.sort_values("adj_projection", ascending=False).groupby("team", sort=False).head(TD_PER_TEAM)
           .sort_values("adj_projection", ascending=False))
    rows, model_only = [], False
    for r in top.itertuples():
        dk = getattr(r, "over_odds", float("nan"))
        fair = fair_american(r.adj_projection)
        priced = pd.notna(getattr(r, "market_prob", np.nan))
        if pd.notna(dk):
            value = (dk / 100 + 1 if dk > 0 else 100 / -dk + 1) * r.adj_projection > 1
            dk_cell = f'<span class="n tdk{" good" if value else ""}">{_odds(dk)}{"<sup>FD</sup>" if _bk(r) == "FD" else ""}</span>'
        else:
            dk_cell = '<span class="n tdk muted">—</span>'
        star = "" if priced else '<sup class="muted" title="No market price yet: model only">*</sup>'
        model_only |= not priced
        inj = f'<span class="inj">{escape(_short_inj(r.injury))}</span>' if r.injury else ""
        rows.append(f"""<li><span class="tdn">{escape(r.player)} <span class="muted">{r.position} · {r.team}</span>{inj}</span>
        <span class="n tdp">{r.adj_projection * 100:.0f}%{star}</span><span class="n tdo">{_odds(fair)}</span>{dk_cell}</li>""")
    # best value: biggest edge at DraftKings' price among everyone with a price (not just the likeliest scorers)
    priced = tds[tds.over_odds.notna() & tds.ev.notna()].sort_values("ev", ascending=False)
    if len(priced) and priced.ev.iloc[0] > 0:
        v = priced.iloc[0]
        other = best.get((v.player, "anytime_td", "Over", 0.5))
        alt = (f" · best anywhere {_odds(other[1])} {escape(str(other[0]))}"
               if other and pd.notna(other[1]) and decimal(float(other[1])) > decimal(float(v.over_odds)) else "")
        tag = "Bet" if v.flag else "Lean"
        value_line = (f'<p class="tdval"><b>Best TD value:</b> {escape(v.player)} ({v.team}) {_bk(v)} {_odds(v.over_odds)}, '
                      f'fair {_odds(fair_american(v.adj_projection))}, edge {_signed_pct(v.ev)} · {tag}{alt}</p>')
    elif len(priced):
        value_line = '<p class="tdval muted">No DraftKings or FanDuel touchdown price beats fair odds in this game right now.</p>'
    else:
        value_line = ""
    note = "Price is DraftKings' unless marked FD (FanDuel pays more). Green = pays more than fair odds (a value bet)."
    if model_only:
        note += " * No market price yet, so the chance is the model's alone; it firms up when DraftKings posts the line."
    return f"""<div class="tds"><h4>Most likely TD scorers</h4><ol class="tdtable">
      <li class="hd"><span>Player</span><span class="n">Chance</span><span class="n">Fair</span><span class="n">Price</span></li>{"".join(rows)}</ol>
      {value_line}<p class="cell-sub">{note}</p></div>"""


def _prop_rows(grp, gi) -> str:
    """Read-only rows: projection, fair line, and the DraftKings line and edge when one was found."""
    rows = []
    for r in grp.itertuples():
        inj = f'<span class="inj">{escape(r.injury)}</span>' if r.injury else ""
        known = pd.notna(r.line)
        fair = (_odds(fair_american(r.a)) if r.stat == "anytime_td" else f"{r.fair_line:g}")
        if known:
            line = "Yes" if r.stat == "anytime_td" else f"{r.line:g}"
            line += f'<span class="cell-sub">{_bk(r)} {escape(r.as_of)}</span>'
            over = _odds(r.over_odds) if pd.notna(r.over_odds) else "—"
            under = _odds(r.under_odds) if pd.notna(r.under_odds) else "—"
            cls = "hit" if r.flag else ("lean" if r.ev > 0 else "no")
            side = "Yes" if r.stat == "anytime_td" else r.best_side
            edge = f'<span class="out {cls}">{side} {_signed_pct(r.ev)}</span>'
        else:
            line, over, under, edge = '<span class="muted">—</span>', "", "", '<span class="muted">No line yet</span>'
        rows.append(f"""<tr>
          <td><b>{escape(r.player)}</b><span class="cell-sub">{r.position} · {r.team} · {STAT_LABEL.get(r.stat, r.stat)}</span>{inj}</td>
          <td class="n" data-l="Proj">{_fmt_proj(r.stat, r.adj_projection)}</td><td class="n" data-l="Fair">{fair}</td>
          <td class="n" data-l="Line">{line}</td><td class="n" data-l="Over">{over}</td><td class="n" data-l="Under">{under}</td><td class="n" data-l="Edge">{edge}</td></tr>""")
    return "".join(rows)


def _card_props(grp, gi, best=None) -> str:
    """Top 5 prop edges for one game, plus every projected prop with a line checker."""
    if grp is None or len(grp) == 0:
        return ""
    priced = grp[grp.line.notna()].sort_values("ev", ascending=False)
    have = priced[priced.ev > 0].head(5)
    if not len(have):
        have = priced.head(5)   # no positive edges: show the closest ones, marked No edge
    if len(have):
        items = []
        for r in have.itertuples():
            if r.stat == "anytime_td":
                what, price = f"Anytime TD · {r.adj_projection * 100:.0f}% to score", r.over_odds
            else:
                price = r.over_odds if r.best_side == "Over" else r.under_odds
                what = f"{r.best_side} {r.line:g} {STAT_LABEL.get(r.stat, r.stat).lower()}"
            tag = "Bet" if r.flag else ("Lean" if r.ev > 0 else "No edge")
            b = (best or {}).get((r.player, r.stat, "Over" if r.stat == "anytime_td" else r.best_side, float(r.line)))
            if b and b[0] not in ("DraftKings", "FanDuel") and pd.notna(price) and decimal(b[1]) > decimal(price):
                what += f" · Best: {b[0]} {_odds(b[1])}"
            if _bk(r) == "FD":
                what = f"{what} · at FanDuel"
            warn = '<span class="inj">Big edge: check the line</span>' if r.ev > 0.15 else ""
            items.append(f"""<li><span class="tdn"><b>{escape(r.player)}</b>{warn}<span class="cell-sub">{escape(what)}</span></span>
            <span class="n tdk">{_odds(price)}</span>
            <span class="n tdp {'good' if r.flag else ('bad' if r.ev <= 0 else '')}">{_signed_pct(r.ev)}</span>
            <span class="tdo">{tag}</span></li>""")
        top = (f'<ol class="tdtable edges"><li class="hd"><span>Bet</span><span class="n">Price</span><span class="n">Edge</span><span></span></li>'
               + "".join(items) + '</ol>')
    else:
        top = '<p class="cell-sub" style="margin:0">No edge · no DraftKings or FanDuel prop lines found yet for this game.</p>'
    return f"""<div class="tds"><h4>Top prop edges</h4>{top}</div>
      <details class="allprops"><summary>All {len(grp)} props</summary>
        <div class="tablebox"><table class="props">
        <thead><tr><th>Player · prop</th><th class="n">Proj.</th><th class="n">Fair</th><th class="n">Line</th>
        <th class="n">Over</th><th class="n">Under</th><th class="n">Edge</th></tr></thead>
        <tbody>{_prop_rows(grp, gi)}</tbody></table></div>
        <p class="cell-sub" style="margin:6px 0 0">Fair is the line where over and under are equally likely (fair odds to score for touchdowns). Compare it with the line in your app.</p>
      </details>"""


def _game_lines_label(raw: str, dk_api: str | None) -> str:
    """Plain-English source of the game's spread / total / moneyline."""
    if dk_api:
        return f"Game lines: DraftKings ({dk_api})"
    raw = (raw or "").strip()
    if not raw or raw.lower().startswith("feed"):
        return "Game lines: sportsbook consensus (no recent DraftKings line)"
    if raw.lower().startswith("draftkings"):
        return "Game lines: " + raw
    return "Game lines: " + raw


def _card_news(n, dk_api=None) -> str:
    if n is None:
        return ""
    label = _game_lines_label(str(n.get("lines", "")), dk_api)
    lines = f'<span class="cell-sub">{escape(label)}</span>'
    status = str(n.get("status", ""))
    return f"""<footer class="card-news"><span class="st {STATUS_CLASS.get(status.lower(), '')}">{escape(status)}</span>
      <p>{escape(str(n.get("note", "")))}{lines}</p></footer>"""


def _ratings_section(ratings, play) -> str:
    t = ratings.table()
    mx = max(1.0, t.net.abs().max())
    rows = []
    for i, r in t.iterrows():
        w = abs(r.net) / mx * 50
        bar = (f'<i class="pos" style="left:50%;width:{w:.1f}%"></i>' if r.net >= 0
               else f'<i class="neg" style="left:{50 - w:.1f}%;width:{w:.1f}%"></i>')
        eo, ed = play["epa"].off.get(r.team, 0), play["epa"].deff.get(r.team, 0)
        so, sd_ = play["sr"].off.get(r.team, 0) * 100, play["sr"].deff.get(r.team, 0) * 100
        rows.append(f"""<tr><td class="n rank">{i + 1}</td><td><b>{r.team}</b> <span class="muted">{NAMES.get(r.team, '')}</span></td>
          <td class="n">{r.off:+.1f}</td><td class="n">{r['def']:+.1f}</td><td class="n"><b>{r.net:+.1f}</b></td>
          <td class="n sep {'good' if eo > 0 else 'bad'}">{eo:+.3f}</td><td class="n {'good' if ed > 0 else 'bad'}">{ed:+.3f}</td>
          <td class="n">{so:+.1f}</td><td class="n">{sd_:+.1f}</td>
          <td class="barcell"><span class="dbar">{bar}</span></td></tr>""")
    return f"""<div class="tablebox"><table class="ratings">
      <thead><tr class="group-head"><th colspan="2"></th><th colspan="3" class="grp">Points per game</th><th colspan="2" class="grp sep">EPA per play</th><th colspan="2" class="grp">Success rate, pct pts</th><th></th></tr>
      <tr><th class="n">#</th><th>Team</th><th class="n">Offense</th><th class="n">Defense</th><th class="n">Net</th>
      <th class="n sep">EPA off</th><th class="n">EPA def</th><th class="n">Success off</th><th class="n">Success def</th><th>Net vs. average</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>
      <p class="note">Every column is better (+) or worse (−) than an average team; on defense, + means the team allows less.
      EPA (expected points added) grades each play by how much it changed a drive's scoring chances, leaving out garbage time.
      Success rate is the share of plays that kept the offense on schedule. The points columns are per game on a neutral field.
      League average is {ratings.mu:.1f} points; home field is worth {ratings.hfa:.1f}.
      A +4 team would be about a 4-point favorite over an average opponent at a neutral site.</p>"""


def _roi_chart(by_season: pd.DataFrame) -> str:
    seasons = list(by_season.index)
    vals = [float(v) for v in by_season.roi]
    W, H, padL, padR, padT, padB = 640, 220, 44, 8, 14, 30
    lim = max(0.15, max(abs(v) for v in vals) * 1.1)
    plotH = H - padT - padB
    y = lambda v: padT + (lim - v) / (2 * lim) * plotH
    bw = (W - padL - padR) / len(vals)
    ticks = [-0.1, -0.05, 0, 0.05, 0.1]
    grid = "".join(
        f'<line x1="{padL}" x2="{W - padR}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="{"axis" if t == 0 else "grid"}"/>'
        f'<text x="{padL - 6}" y="{y(t) + 4:.1f}" text-anchor="end" class="tick">{t * 100:+.0f}%</text>'
        for t in ticks if abs(t) <= lim)
    bars, labels = [], []
    for i, (s, v) in enumerate(zip(seasons, vals)):
        x = padL + i * bw + bw * 0.18
        top, bot = (y(v), y(0)) if v >= 0 else (y(0), y(v))
        bars.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw * 0.64:.1f}" height="{max(1, bot - top):.1f}" '
                    f'class="{"bpos" if v >= 0 else "bneg"}"><title>{s}: {v * 100:+.1f}% ROI</title></rect>')
        lbl = str(s)[2:] if len(seasons) > 10 else str(s)
        labels.append(f'<text x="{x + bw * 0.32:.1f}" y="{H - 10}" text-anchor="middle" class="tick">\'{lbl}</text>')
    return (f'<div class="chartbox"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Return on flagged bets by season">'
            f'{grid}{"".join(bars)}{"".join(labels)}</svg></div>')


def _backtest_section(acc, rec, summarize) -> str:
    bets = rec[rec.ev >= C.MIN_EV]
    by_m = bets.groupby("market").apply(summarize, include_groups=False)
    by_s = bets.groupby("season").apply(summarize, include_groups=False)
    allr = summarize(bets)
    first, last = int(acc.season.min()), int(acc.season.max())
    mrows = "".join(
        f"""<tr><td><b>{MARKET_LABEL[m]}</b></td><td class="n">{int(r.bets):,}</td><td class="n">{_pct(r.win_pct, 1)}</td>
        <td class="n">{r.units:+.1f}</td><td class="n {'good' if r.roi > 0 else 'bad'}">{_signed_pct(r.roi)}</td></tr>"""
        for m, r in by_m.iterrows())
    mrows += (f"""<tr class="total"><td><b>All markets</b></td><td class="n">{int(allr.bets):,}</td><td class="n">{_pct(allr.win_pct, 1)}</td>
        <td class="n">{allr.units:+.1f}</td><td class="n {'good' if allr.roi > 0 else 'bad'}">{_signed_pct(allr.roi)}</td></tr>""")
    se = 1 / np.sqrt(max(1, allr.bets))
    mae_mkt, mae_model, mae_blend = acc.mae_market.mean(), acc.mae_model.mean(), acc.mae_blend.mean()
    w_now = acc.w_spread.iloc[-1]
    return f"""
    <div class="callout">
      <p><b>The short version.</b> Over {first}–{last}, bets the model flagged returned {_signed_pct(allr.roi)} per dollar at closing odds.
      Random luck alone moves a sample this size by about ±{se * 100:.0f}%, so the model has not shown a reliable edge.
      Moneylines did a little better than break-even, spreads about broke even, and totals lost roughly the bookmaker's cut.</p>
    </div>
    <div class="two">
      <div><h3>Results by bet type</h3>
        <div class="tablebox"><table>
          <thead><tr><th>Market</th><th class="n">Bets</th><th class="n">Won</th><th class="n">Units</th><th class="n">ROI</th></tr></thead>
          <tbody>{mrows}</tbody></table></div>
        <p class="note">One unit is one dollar risked per bet. Pushes count as bets but not as wins or losses.</p>
      </div>
      <div><h3>Return by season</h3>{_roi_chart(by_s)}
        <p class="note">Green seasons made money, red seasons lost. The swings are the noise you should expect week to week.</p></div>
    </div>
    <h3>How close the lines are</h3>
    <p>Average miss on the final margin: market <b class="n">{mae_mkt:.2f}</b> points, model alone <b class="n">{mae_model:.2f}</b>,
    blended projection <b class="n">{mae_blend:.2f}</b>. The market is the more accurate number, which is why the model
    currently gives its own spread only <b>{w_now:.0%}</b> weight and leans on the line for the rest.</p>"""


def _props_section(props: pd.DataFrame | None) -> str:
    if props is None or props.empty:
        return ""
    rows = []
    for p in props.itertuples():
        ex = '<span class="ex">Example</span>' if str(p.player).lower().startswith("example") else ""
        proj = f"{p.adj_projection:.0%}" if p.stat == "anytime_td" else f"{p.adj_projection:.1f}"
        line = "Scores" if p.stat == "anytime_td" else f"{p.line}"
        verdict = (f'<b>{p.best_side}</b> · ${p.stake:,.0f}' if p.flag
                   else (f"Lean {p.best_side}" if p.ev > 0 else "Pass"))
        rows.append(f"""<tr><td><b>{escape(str(p.player))}</b>{ex}<span class="cell-sub">{p.team}</span></td>
          <td>{STAT_LABEL.get(p.stat, p.stat)}</td><td class="n">{proj}<span class="cell-sub">×{p.script_factor:.2f} game script</span></td>
          <td class="n">{line}</td><td class="n">{_pct(p.p_over, 1)}</td><td>{verdict}</td>
          <td class="n {'good' if p.ev > 0 else 'bad'}">{_signed_pct(p.ev)}</td></tr>""")
    return f"""<section id="props"><h2>Player props</h2>
      <p class="lede">Your projections, scaled by how many points the model expects each team to score this week.</p>
      <div class="tablebox"><table>
      <thead><tr><th>Player</th><th>Stat</th><th class="n">Adjusted projection</th><th class="n">Line</th>
      <th class="n">Chance of over</th><th>Call</th><th class="n">Edge</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>
      <p class="note">Rows marked Example are placeholders in props_example.csv. Replace them with real players and your book's lines.</p></section>"""


STATUS_CLASS = {"ok": "good", "watch": "watch", "updated": "upd", "pulled": "bad", "started": "muted"}


def _news_section(news: pd.DataFrame | None, checked: str | None) -> str:
    """Per-game QB, line and injury notes from news.csv (game,status,qbs,note)."""
    if news is None or news.empty:
        return ""
    has_lines = "lines" in news.columns
    rows = "".join(
        f"""<tr><td><b>{escape(str(r.game))}</b></td><td>{escape(str(r.qbs))}</td>
        <td><span class="st {STATUS_CLASS.get(str(r.status).lower(), '')}">{escape(str(r.status))}</span></td>
        {f'<td>{escape(str(r.lines))}</td>' if has_lines else ''}<td>{escape(str(r.note))}</td></tr>"""
        for r in news.fillna("").itertuples())
    when = f" · {escape(checked)}" if checked else ""
    return f"""<h3>News check{when}</h3>
    <p class="lede">Starting quarterbacks, injuries and line moves checked against current reports.
    "Updated" means the model was rerun with a correction for that game. Lines are DraftKings' when two sources from the last
    12 hours confirm them; otherwise the model uses the nflverse consensus feed.</p>
    <div class="tablebox"><table>
      <thead><tr><th>Game</th><th>Starting QBs</th><th>Status</th>{'<th>Lines used</th>' if has_lines else ''}<th>Notes</th></tr></thead>
      <tbody>{rows}</tbody></table></div>"""


def _fmt_proj(stat, x) -> str:
    if stat == "anytime_td":
        return f"{x * 100:.0f}%"
    if stat in ("receptions", "pass_tds"):
        return f"{x:.1f}"
    return f"{x:.0f}"


def _auto_props_section(props: pd.DataFrame | None, lines_checked: str | None) -> str:
    """Model projections for every starter, a line checker per row, and edges where DraftKings lines are known."""
    if props is None or props.empty:
        return ""
    have = props[props.line.notna()]
    flagged = have[have.flag].sort_values("ev", ascending=False)
    if len(flagged):
        bet_rows = "".join(
            f"""<tr><td><b>{escape(r.player)}</b><span class="cell-sub">{r.team} vs {r.opp} · {r.position}</span></td>
            <td>{STAT_LABEL.get(r.stat, r.stat)}</td>
            <td class="n">{'Yes' if r.stat == 'anytime_td' else f'{r.best_side} {r.line:g}'}</td>
            <td class="n">{_odds(r.over_odds if r.best_side == 'Over' else r.under_odds)}</td>
            <td class="n">{_fmt_proj(r.stat, r.adj_projection)}</td>
            <td class="n">{_pct(r.p_over if r.best_side == 'Over' else 1 - r.p_over, 1)}</td>
            <td class="n good">{_signed_pct(r.ev)}</td><td class="n">${r.stake:,.0f}</td>
            <td><span class="cell-sub">{escape(r.source)} · {escape(r.as_of)}</span></td></tr>"""
            for r in flagged.itertuples())
        bets = f"""<div class="tablebox"><table>
          <thead><tr><th>Player</th><th>Prop</th><th class="n">Bet</th><th class="n">Odds</th><th class="n">Projection</th>
          <th class="n">Model chance</th><th class="n">Edge</th><th class="n">Stake</th><th>Line from</th></tr></thead>
          <tbody>{bet_rows}</tbody></table></div>"""
    else:
        bets = (f'<p class="lede">No prop clears the {C.MIN_EV:.0%} bar yet. '
                f'DraftKings lines found so far: {len(have)} of {len(props)} props. '
                'Use the checker below with the lines in your app.</p>')

    games = []
    for gi, (game, grp) in enumerate(props.groupby("game", sort=False)):
        rows = []
        for i, r in enumerate(grp.itertuples()):
            rid = f"p{gi}_{i}"
            inj = f'<span class="inj">{escape(r.injury)}</span>' if r.injury else ""
            known = pd.notna(r.line)
            line_val = "" if r.stat == "anytime_td" else (f"{r.line:g}" if known else f"{r.fair_line:g}")
            oo = f"{int(r.over_odds):+d}" if known and pd.notna(r.over_odds) else ("" if r.stat == "anytime_td" else "-110")
            uo = f"{int(r.under_odds):+d}" if known and pd.notna(r.under_odds) else ("" if r.stat == "anytime_td" or known else "-110")
            fair = (_odds(fair_american(r.a)) if r.stat == "anytime_td" else f"{r.fair_line:g}")
            src = f'<span class="cell-sub">{_bk(r)} {escape(r.as_of)}</span>' if known else ""
            line_cell = ('<span class="muted">Scores</span>' if r.stat == "anytime_td" else
                         f'<input id="{rid}_l" class="pin" inputmode="decimal" value="{line_val}" aria-label="Line">')
            under_cell = ('' if r.stat == "anytime_td" else
                          f'<input id="{rid}_u" class="pin" inputmode="text" value="{uo}" aria-label="Under odds">')
            rows.append(f"""<tr class="pr" data-d="{r.dist}" data-a="{r.a}" data-b="{r.b}">
              <td><b>{escape(r.player)}</b><span class="cell-sub">{r.position} · {r.team}</span>{inj}</td>
              <td>{STAT_LABEL.get(r.stat, r.stat)}</td>
              <td class="n">{_fmt_proj(r.stat, r.adj_projection)}</td><td class="n">{fair}</td>
              <td class="n">{line_cell}{src}</td>
              <td class="n"><input id="{rid}_o" class="pin" inputmode="text" value="{oo}" aria-label="Over odds"></td>
              <td class="n">{under_cell}</td>
              <td class="n out">—</td></tr>""")
        n_dk = int(grp.line.notna().sum())
        games.append(f"""<details class="pg"><summary><b>{escape(game)}</b>
          <span class="muted">{grp.player.nunique()} players · {len(grp)} props{f' · {n_dk} DraftKings lines' if n_dk else ''}</span></summary>
          <div class="tablebox"><table class="props">
          <thead><tr><th>Player</th><th>Prop</th><th class="n">Projection</th><th class="n">Fair line</th>
          <th class="n">Line</th><th class="n">Over odds</th><th class="n">Under odds</th><th class="n">Best side</th></tr></thead>
          <tbody>{''.join(rows)}</tbody></table></div></details>""")
    when = f" DraftKings prop lines last checked {escape(lines_checked)}." if lines_checked else ""
    return f"""<section id="props"><h2>Player props</h2>
      <p class="lede">Projections for every starting QB and the main backs, receivers and tight ends, from their recent games,
      scaled by how many points the model expects their team to score. Type the line and odds from your app to see the edge.</p>
      <p class="note">Only DraftKings prop lines published in the last 24 hours are used, and each shows when it was published.{when} Prop projections are new and haven't been
      tested against past seasons the way the game model has, and they don't adjust for the opponent's defense yet, so treat edges as a starting point.</p>
      <h3>Every player</h3>
      <p class="note" style="margin-bottom:10px">Fair line is the line where over and under are equally likely; for touchdowns it's fair odds to score.
      Players on IR and players listed Out or Doubtful are left out.</p>
      {''.join(games)}
      <script>{PROPS_JS}</script></section>"""


PROPS_JS = r"""
(function(){
function erf(x){var t=1/(1+0.3275911*Math.abs(x));var y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);return x>=0?y:-y;}
function ncdf(x,m,s){return 0.5*(1+erf((x-m)/(s*Math.SQRT2)));}
function pcdf(k,l){if(k<0)return 0;var p=Math.exp(-l),s=p;for(var i=1;i<=k;i++){p*=l/i;s+=p;}return s;}
function ppmf(k,l){if(k<0)return 0;var p=Math.exp(-l);for(var i=1;i<=k;i++){p*=l/i;}return p;}
function dec(a){a=parseFloat(a);if(!isFinite(a)||a===0||Math.abs(a)<100)return NaN;return a>0?1+a/100:1+100/Math.abs(a);}
function probs(d,a,b,line){
  if(d==='td')return [a,0,1-a];
  if(d==='normal'){var o=1-ncdf(line,a,b);return [o,0,1-o];}
  var k=Math.floor(line);
  if(line===k)return [1-pcdf(k,a),ppmf(k,a),pcdf(k-1,a)];
  return [1-pcdf(k,a),0,pcdf(k,a)];
}
function ev(p,o){var x=dec(o);return isFinite(x)?p[0]*(x-1)-p[2]:NaN;}
function run(tr){
  var d=tr.dataset.d,a=parseFloat(tr.dataset.a),b=parseFloat(tr.dataset.b);
  var li=tr.querySelector('input[id$="_l"]'),oi=tr.querySelector('input[id$="_o"]'),ui=tr.querySelector('input[id$="_u"]');
  var line=d==='td'?0.5:parseFloat(li&&li.value);
  var out=tr.querySelector('.out');
  if(!isFinite(line)){out.textContent='—';out.className='n out';return;}
  var p=probs(d,a,b,line),q=[p[2],p[1],p[0]];
  var eo=ev(p,oi.value),eu=ui?ev(q,ui.value):NaN;
  var best=null;
  if(isFinite(eo))best=['Over',eo,p[0]];
  if(isFinite(eu)&&(!best||eu>best[1]))best=['Under',eu,q[0]];
  if(!best){out.textContent=(p[0]*100).toFixed(0)+'% over';out.className='n out';return;}
  var lbl=d==='td'?'Yes':best[0];
  out.textContent=lbl+' '+(best[1]>=0?'+':'')+(best[1]*100).toFixed(1)+'%';
  out.className='n out '+(best[1]>=0.03?'hit':(best[1]>0?'lean':'no'));
  out.title='Model chance '+(best[2]*100).toFixed(1)+'%';
}
var rows=document.querySelectorAll('tr.pr');
for(var i=0;i<rows.length;i++){(function(tr){run(tr);tr.addEventListener('input',function(){run(tr);});})(rows[i]);}
})();
"""


# ------------------------------------------------------------------ page
CSS = """
:root{--bg:#F1F4F2;--surface:#FFFFFF;--ink:#14211A;--muted:#5C6C63;--line:#D6DED9;--chip:#E6ECE8;
--accent:#17663F;--flag:#C99700;--flag-soft:#FBF1CF;--good:#17663F;--bad:#B23A2E;
--display:"Saira Condensed","Arial Narrow",Arial,sans-serif;--body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0F1612;--surface:#17211B;--ink:#E5EEE8;--muted:#94A69B;
--line:#2A3830;--chip:#223028;--accent:#4FC184;--flag:#F0C33C;--flag-soft:#3A3115;--good:#4FC184;--bad:#EE7A6C;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0F1612;--surface:#17211B;--ink:#E5EEE8;--muted:#94A69B;--line:#2A3830;--chip:#223028;
--accent:#4FC184;--flag:#F0C33C;--flag-soft:#3A3115;--good:#4FC184;--bad:#EE7A6C;color-scheme:dark}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.55}
.wrap{max-width:1080px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.mast{background:var(--ink);color:var(--bg)}
.mast .wrap{padding-block:28px 22px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:12px 24px}
.kicker{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;opacity:.7;margin:0 0 4px}
h1{font-family:var(--display);font-weight:800;font-size:clamp(34px,6vw,52px);line-height:.95;text-transform:uppercase;margin:0;letter-spacing:.01em;text-wrap:balance}
.mast-stats{display:flex;gap:24px;flex-wrap:wrap}
.mast-stats div{display:flex;flex-direction:column}
.mast-stats b{font-family:var(--display);font-size:30px;line-height:1}
.mast-stats span{font-size:12px;opacity:.7}
nav.toc{border-bottom:1px solid var(--line);background:var(--surface);position:sticky;top:0;z-index:10}
nav.toc .wrap{display:flex;gap:4px;overflow-x:auto;padding-block:0;scrollbar-width:none}
nav.toc .wrap::-webkit-scrollbar{display:none}
nav.toc a{color:var(--muted);text-decoration:none;font-weight:600;font-size:13px;white-space:nowrap;text-transform:uppercase;letter-spacing:.06em;
padding:13px 12px 11px;border-bottom:3px solid transparent;margin-bottom:-1px}
nav.toc a:hover{color:var(--ink)}
nav.toc a:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
nav.toc a[aria-selected="true"]{color:var(--accent);border-bottom-color:var(--accent)}
html.tabs main>section{display:none}
html.tabs main>section.active{display:block}
main{padding-block:12px 56px}
section{padding-top:32px}
h2{font-family:var(--display);font-weight:700;font-size:28px;text-transform:uppercase;letter-spacing:.02em;margin:0 0 6px;text-wrap:balance}
h3{font-family:var(--display);font-weight:700;font-size:19px;text-transform:uppercase;letter-spacing:.03em;margin:22px 0 8px}
.lede{color:var(--muted);margin:0 0 16px;max-width:68ch}
p{max-width:72ch}
.note{font-size:13px;color:var(--muted);margin:8px 0 0}
.muted{color:var(--muted)}
.n{font-family:var(--mono);font-variant-numeric:tabular-nums}
.good{color:var(--good)} .bad{color:var(--bad)}
.tablebox{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:600;white-space:nowrap}
th.n,td.n{text-align:right;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
tr.total td{border-top:2px solid var(--line)}
.cell-sub{display:block;font-family:var(--body);font-size:12px;color:var(--muted);font-weight:400}
.cards{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:6px;display:flex;flex-direction:column;min-width:0}
.card.has-bet{border-color:var(--flag);box-shadow:inset 0 3px 0 var(--flag)}
.card.final{border-color:var(--line);box-shadow:none}
.card.final .pts b{color:var(--ink)}
.final-lbl{color:var(--ink);text-transform:uppercase;letter-spacing:.06em;font-size:12px}
.final-head{margin:18px 0 0;color:var(--muted)}
.card-head{padding:14px 16px 10px;border-bottom:1px solid var(--line)}
.teams{display:flex;flex-direction:column;gap:4px}
.team{display:grid;grid-template-columns:52px minmax(0,1fr) auto;column-gap:10px;align-items:baseline}
.abbr{font-family:var(--display);font-weight:800;font-size:24px;line-height:1.1}
.nm{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.qb{grid-column:2;grid-row:2;font-size:12px;color:var(--muted)}
.pts{grid-column:3;grid-row:1/span 2;font-size:22px;font-weight:600;align-self:center}
.when{margin-top:8px;font-size:13px;color:var(--muted)}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.tags:empty{display:none}
.tag{font-size:11px;font-weight:600;letter-spacing:.04em;padding:2px 7px;border-radius:3px;background:var(--chip);color:var(--muted)}
.lines{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));flex:1}
.line{padding:12px 14px;border-right:1px solid var(--line);min-width:0;display:flex;flex-direction:column}
.line:last-child{border-right:0}
.line h4{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
dl{margin:0;display:grid;grid-template-columns:auto auto;gap:2px 8px;font-size:13px}
dt{color:var(--muted)} dd{margin:0;text-align:right;white-space:nowrap}
.lean{margin-top:auto;padding-top:10px;font-size:12.5px;font-weight:600}
.lean .n{font-weight:500}
.lean.bet{color:var(--ink)}
.lean.bet::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--flag);margin-right:6px;vertical-align:1px}
.lean.pass{color:var(--muted);font-weight:500}
.wp{margin-bottom:8px}
.wp-bar{height:8px;border-radius:4px;background:var(--accent);overflow:hidden}
.wp-bar i{display:block;height:100%;background:var(--muted);opacity:.45}
.wp-lbl{display:flex;justify-content:space-between;font-size:12px;margin-top:3px}
.ratings td{vertical-align:middle}
.rank{color:var(--muted);width:36px}
.barcell{width:22%;min-width:120px}
.sep{border-left:1px solid var(--line)}
.group-head th{border-bottom:0;padding-block:10px 0}
.grp{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);text-align:center}
.dbar{position:relative;display:block;height:10px;background:var(--chip);border-radius:2px}
.dbar::after{content:"";position:absolute;left:50%;top:-3px;width:1px;height:16px;background:var(--muted)}
.dbar i{position:absolute;top:0;height:10px;border-radius:2px}
.dbar i.pos{background:var(--good)} .dbar i.neg{background:var(--bad)}
.callout{background:var(--flag-soft);border-left:4px solid var(--flag);border-radius:4px;padding:12px 16px;margin:6px 0 4px}
.callout p{margin:0}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}
.chartbox{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px;overflow-x:auto}
.chartbox svg{display:block;width:100%;height:auto}
.grid{stroke:var(--line);stroke-width:1} .axis{stroke:var(--muted);stroke-width:1}
.tick{fill:var(--muted);font-family:var(--mono);font-size:11px}
.bpos{fill:var(--good)} .bneg{fill:var(--bad)}
.st{font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:2px 7px;border-radius:3px;background:var(--chip);color:var(--muted);white-space:nowrap}
.st.good{color:var(--good)} .st.bad{color:var(--bad)} .st.watch{color:var(--flag);background:var(--flag-soft)} .st.upd{color:var(--accent)}
.pg{background:var(--surface);border:1px solid var(--line);border-radius:6px;margin-bottom:8px}
.pg summary{cursor:pointer;padding:10px 14px;display:flex;flex-wrap:wrap;gap:4px 12px;align-items:baseline}
.pg summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.pg .tablebox{border:0;border-top:1px solid var(--line);border-radius:0 0 6px 6px}
.pin{width:64px;font:inherit;font-family:var(--mono);font-size:13px;text-align:right;padding:3px 6px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink)}
.pin:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.inj{display:block;font-size:11.5px;color:var(--flag);font-weight:600}
.out.hit{color:var(--good);font-weight:700} .out.lean{color:var(--ink)} .out.no{color:var(--muted)}
table.props td{vertical-align:middle}
.card-extra{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);border-top:1px solid var(--line)}
.card-extra>.tds{border-top:0}
.card-extra>.tds+div,.card-extra>.tds+.tds{border-left:1px solid var(--line)}
.card-extra details.allprops{grid-column:1/-1;border-top:1px solid var(--line);padding:8px 14px 10px}
.allprops summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:13px}
.allprops summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.allprops .tablebox{margin-top:8px}
@media (max-width:640px){.card-extra{grid-template-columns:minmax(0,1fr)}.card-extra>.tds+.tds{border-left:0;border-top:1px solid var(--line)}}
.tds{padding:10px 14px 8px;border-top:1px solid var(--line)}
.tds h4{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.tds ol{margin:0;padding:0;list-style:none;display:grid;gap:3px;counter-reset:td}
.tds li{display:grid;grid-template-columns:18px minmax(0,1fr) auto 56px;gap:8px;align-items:baseline;font-size:13px;counter-increment:td}
.tds li::before{content:counter(td);color:var(--muted);font-family:var(--mono);font-size:12px}
.tdn{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tdn .inj{display:inline;margin-left:4px}
.tdp{font-weight:600} .tdo{color:var(--muted);text-align:right}
.tdtable li{grid-template-columns:18px minmax(0,1fr) 46px 50px 50px}
.tdtable li.hd{counter-increment:none;font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);padding-bottom:2px;border-bottom:1px solid var(--line)}
.tdtable li.hd::before{content:""}
.tdtable:not(.edges) li{align-items:start}
.tdtable:not(.edges) .tdn{white-space:normal;overflow:visible}
.tdtable:not(.edges) .tdn>.muted{white-space:nowrap}
.tdtable:not(.edges) .tdn .inj{display:block;margin:1px 0 0;font-size:11px;line-height:1.3}
.tdtable .tdp,.tdtable .tdo,.tdtable .tdk,.tdtable li.hd .n{text-align:right}
.tdtable.edges li{grid-template-columns:18px minmax(0,1fr) 50px 54px 52px;align-items:center}
.tdtable.edges .tdn{white-space:normal}
@media (max-width:640px){.tdtable.edges li{grid-template-columns:16px minmax(0,1fr) 44px 50px 36px;gap:6px}}
.tdtable.edges .tdn .cell-sub{margin:0}
.tdtable.edges .tdn .inj{display:block;margin:0}
.card-inj{padding:8px 14px;border-top:1px solid var(--line);font-size:13px;display:grid;gap:3px}
.card-inj h4{margin:0 0 2px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.card-inj .cell-sub{margin:2px 0 0}
.move{margin-top:4px}
.lean .cell-sub{display:block;margin-top:2px}
.beta{display:inline-block;vertical-align:middle;font-family:var(--body);font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:3px 8px;border-radius:3px;background:var(--flag);color:#14211A;margin-left:8px;position:relative;top:-4px}
.beta-note{font-size:13px;color:var(--muted);background:var(--flag-soft);border-left:3px solid var(--flag);border-radius:4px;padding:8px 12px;margin:14px 0 0;max-width:none}
.tdk{font-weight:600} .tdk.good{color:var(--good,#17663F)}
.tds .cell-sub{margin-top:6px}
.card-news{display:flex;gap:10px;align-items:flex-start;padding:10px 14px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
.card-news p{margin:0;max-width:none}
.card-news .cell-sub{margin-top:2px}
.best{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
.best h3{margin-top:4px}
.leanstrip{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:4px 0 18px}
.leanstrip>div{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 14px}
.leanstrip span{display:block;font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:600}
.leanstrip b{display:block;font:700 26px/1.1 var(--display);margin:2px 0}
.leanstrip em{font-style:normal;font-size:12.5px;color:var(--muted)} .leanstrip i{font-style:normal;font-weight:600}
#leans .final-head{margin-top:24px}
.tdval{font-size:12.5px;margin:8px 0 2px;line-height:1.4} .tdval b{font-weight:600}
.books{white-space:nowrap} .books.good{color:var(--good)} .books.bad{color:var(--bad)}
@media (max-width:520px){#leans .hide-sm{display:none}}
.empty{background:var(--surface);border:1px dashed var(--line);border-radius:6px;padding:14px 16px;margin:0;color:var(--muted);font-size:14px}
.updated{font-size:13px;color:var(--muted);margin:0 0 14px}
details.more{margin-top:4px}
details.more summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:14px;padding:6px 0}
details.more summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:860px){.best{grid-template-columns:minmax(0,1fr)}}
.ex{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--flag);background:var(--flag-soft);padding:1px 5px;border-radius:3px;margin-left:6px}
.glossary{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px 28px}
.glossary div b{display:block;font-family:var(--display);text-transform:uppercase;letter-spacing:.03em;font-size:16px}
.glossary div p{margin:2px 0 0;color:var(--muted);font-size:14px}
footer{border-top:1px solid var(--line);margin-top:40px;padding-block:16px 32px;font-size:12.5px;color:var(--muted)}
a{color:var(--accent)}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:860px){.two{grid-template-columns:minmax(0,1fr)}}
@media (max-width:520px){.lines{grid-template-columns:1fr}.line{border-right:0;border-bottom:1px solid var(--line)}.line:last-child{border-bottom:0}
 .barcell{min-width:100px}}
@media print{nav.toc{display:none}.card{break-inside:avoid}}
"""


TABS_JS = """
(function(){
  var root=document.documentElement, links=[].slice.call(document.querySelectorAll('nav.toc a[data-tab]'));
  var panels=links.map(function(a){return document.getElementById(a.dataset.tab)}).filter(Boolean);
  if(!links.length||!panels.length) return;
  root.classList.add('tabs');
  panels.forEach(function(p){p.setAttribute('role','tabpanel')});
  function show(id, scroll){
    if(!document.getElementById(id)) id=links[0].dataset.tab;
    panels.forEach(function(p){p.classList.toggle('active', p.id===id)});
    links.forEach(function(a){var on=a.dataset.tab===id; a.setAttribute('aria-selected', on); a.tabIndex=on?0:-1;
      if(on){var bar=a.parentNode, l=a.offsetLeft-bar.offsetLeft, r=l+a.offsetWidth;
        if(l<bar.scrollLeft) bar.scrollLeft=l-8; else if(r>bar.scrollLeft+bar.clientWidth) bar.scrollLeft=r-bar.clientWidth+8;}});
    if(scroll){var nav=document.querySelector('nav.toc'), main=document.querySelector('main');
      var top=main.getBoundingClientRect().top+window.pageYOffset-nav.offsetHeight;
      if(window.pageYOffset>top) window.scrollTo(0, top);}
    if(scroll){try{history.replaceState(null,'','#'+id)}catch(e){}}
  }
  links.forEach(function(a,i){
    a.addEventListener('click',function(e){e.preventDefault(); show(a.dataset.tab, true)});
    a.addEventListener('keydown',function(e){
      var j = e.key==='ArrowRight'?i+1 : e.key==='ArrowLeft'?i-1 : null;
      if(j===null) return; e.preventDefault(); j=(j+links.length)%links.length; links[j].focus(); show(links[j].dataset.tab, false);
    });
  });
  window.addEventListener('hashchange',function(){show(location.hash.slice(1), true)});
  show((location.hash||'').slice(1), false);
  if(location.hash){window.addEventListener('load',function(){[0,150,400].forEach(function(t){setTimeout(function(){window.scrollTo(0,0)},t)})})}
})();
"""


def render(*, season, week, sm, games, ratings, play, acc, rec, props, summarize, data_through,
           news=None, checked=None, auto_props=None, lines_checked=None, injuries=None, clv_log=None,
           prop_bt=None, market_info=None) -> str:
    # finished games drop out of the bet counts and best bets, and their cards move to the bottom with the final score
    # (from the nflverse feed, or from the "Final: ..." news note when the feed hasn't posted the score yet)
    if auto_props is not None and not auto_props.empty and "injury" in auto_props:
        auto_props = auto_props.assign(injury=auto_props.injury.fillna("").astype(str))
    nb = {}
    if news is not None and not news.empty:
        for r in news.fillna("").to_dict("records"):
            nb[str(r.get("game", "")).strip()] = r
    games = [dict(it, g=_final_from_news(it["g"], nb.get(f"{it['g'].away_team} @ {it['g'].home_team}", {}).get("note"))
                  if not _is_final(it["g"]) and _final_from_news(it["g"], nb.get(f"{it['g'].away_team} @ {it['g'].home_team}", {}).get("note")) is not None
                  else it["g"]) for it in games]
    live = [it for it in games if not _is_final(it["g"])]
    done = [it for it in games if _is_final(it["g"])]
    # bets and leans only for games that haven't kicked off; started games keep their cards and frozen pre-game props
    upcoming = [it for it in live if not _started(it["g"])]
    started_keys = {f"{it['g'].away_team} @ {it['g'].home_team}" for it in games if _started(it["g"]) or _is_final(it["g"])}
    open_props = (auto_props[~auto_props.game.isin(started_keys)] if auto_props is not None and not auto_props.empty
                  else auto_props)
    n_bets = sum(1 for it in upcoming for o in it["opts"] if o["ev"] >= C.MIN_EV)
    n_props = int(open_props.flag.sum()) if open_props is not None and not open_props.empty else 0
    news_by_game = {}
    if news is not None and not news.empty:
        for r in news.fillna("").to_dict("records"):
            news_by_game[str(r.get("game", "")).strip()] = r
    td_by_game = {}
    if auto_props is not None and not auto_props.empty:
        td = auto_props[auto_props.stat == "anytime_td"].sort_values("adj_projection", ascending=False)
        td_by_game = {g: grp for g, grp in td.groupby("game", sort=False)}
    props_by_game = {}
    if auto_props is not None and not auto_props.empty:
        props_by_game = {g: grp for g, grp in auto_props.groupby("game", sort=False)}
    key = lambda it: f"{it['g'].away_team} @ {it['g'].home_team}"
    card = lambda gi, it: _game_card(it, news_by_game.get(key(it)), td_by_game.get(key(it)), props_by_game.get(key(it)), gi, injuries, market_info)
    cards = "".join(card(gi, it) for gi, it in enumerate(live))
    if done:
        cards += '<h3 class="final-head">Final</h3>' + "".join(card(len(live) + gi, it) for gi, it in enumerate(done))
    tabs = [("bets", "Best bets"), ("games", "Games"), ("leans", "Leans"), ("results", "Results"), ("ratings", "Ratings"),
            ("backtest", "Track record"), ("how", "Glossary")]
    toc = [f'<a href="#{k}" role="tab" data-tab="{k}" aria-controls="{k}">{v}</a>' for k, v in tabs]
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/Chicago"))
    generated = now.strftime("%b %-d, %Y at %-I:%M %p CT")
    updated = now.strftime("%-I:%M %p")
    dt = pd.Timestamp(data_through).strftime("%b %-d")
    checked_line = (f'<p class="updated">Injuries, starting QBs and lines checked {escape(checked)}. '
                    f'Lines move, so confirm the price in your app before betting.</p>') if checked else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>The Rat Report</title>
<script>try{{var t=localStorage.getItem("rr_theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}}catch(e){{}}</script>
<meta name="color-scheme" content="light dark">
{FAVICON_TAG}
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@700;800&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}{UI_CSS}</style></head>
<body>
<div class="mast"><div class="wrap">
  <div class="brand"><span class="logo">{LOGO_HTML}</span><div class="brand-meta"><h1 class="sr-only">The Rat Report</h1><p class="kicker">Week {week} · {season} NFL season</p><span class="beta">Beta</span></div></div>
  <div class="mast-stats">
    <div><b>{n_bets}</b><span>game bets</span></div>
    <div><b>{n_props}</b><span>prop bets</span></div>
    <div><b>{updated}</b><span>updated (CT)</span></div>
    <button type="button" class="theme-btn" aria-label="Switch between light and dark mode" title="Light / dark mode"><span aria-hidden="true"></span></button>
  </div>
</div></div>
<nav class="toc" aria-label="Sections"><div class="wrap" role="tablist">{''.join(toc)}</div></nav>
<main class="wrap">
  <p class="beta-note"><b>Beta.</b> The model is still being tested: every flagged bet is tracked against the closing line under Track record, and features are kept only when they test better on past games. Treat picks as experimental.</p>
  {_now_section(upcoming, open_props, clv_log, (market_info or {}).get('dk_used', {}), now, sm, _why_ctx(ratings, play, injuries, market_info))}
    {checked_line}
    <p class="note">Prop prices are DraftKings' or FanDuel's (DK/FD), whichever pays more for the same bet. Prop projections adjust for the opponent's defense and expected pace, and blend with the DraftKings line; their replay on past weeks is under Track record.{(' Prop lines last checked ' + escape(lines_checked) + '.') if lines_checked else ''}</p>
    <p class="note">Edge is the expected profit per dollar bet: +5% means about $5 back per $100 over many bets. Bets need at least a {C.MIN_EV:.0%} edge; smaller positive edges are leans.
    Hover or long-press a win % to see the break-even rate.</p></section>
  {_leans_section(upcoming, open_props, clv_log)}
  {_results_section(clv_log)}
  <section id="games"><h2>Games</h2>
    <p class="lede">Projected score beside each team. "Pure model" is the model alone; "Projection" blends it with the market.
    A gold bar marks a game with a bet. Finished games move to the bottom with the final score. Each card has the most likely touchdown scorers, the top prop edges, every projected prop, and the latest news.</p>
    <div class="cards">{cards}</div></section>
  {'' if auto_props is not None and not auto_props.empty else _props_section(props)}
  <section id="ratings"><h2>Power ratings</h2>
    <p class="lede">Ranked by points-based net rating, built from the last two-plus seasons with recent games weighted most.
    Play-by-play efficiency (EPA and success rate) sits alongside and feeds the projections too.</p>
    <details class="more"><summary>Show all 32 teams</summary>{_ratings_section(ratings, play)}</details></section>
  <section id="backtest"><h2>Track record</h2>
    <p class="lede">Live results first, then the game and prop models replayed on past seasons using only information available at the time.</p>
    {_clv_section(clv_log)}
    {_prop_bt_section(prop_bt)}
    <h3>Game model, past seasons</h3>
    <p class="lede">Each season replayed week by week, graded against closing lines.</p>
    {_backtest_section(acc, rec, summarize)}</section>
  <section id="how"><h2>Glossary</h2>
    <div class="glossary">
      <div><b>Spread</b><p>"CHI −2.5" means the Bears are projected to win by 2.5. Plus numbers are underdogs.</p></div>
      <div><b>Market</b><p>The betting line: DraftKings' number when we have a recent one, otherwise the consensus line across sportsbooks. It reflects everything bettors and bookmakers know, including injuries and news.</p></div>
      <div><b>Pure model</b><p>The model's own number, built only from team ratings, play-by-play efficiency, quarterback ratings, injuries, rest, travel and weather, without looking at the betting line.</p></div>
      <div><b>QB rating</b><p>Expected points added per dropback (passes, sacks and scrambles), recent games weighted most. QBs with little film are assumed to be typical backups until they show otherwise.</p></div>
      <div><b>Aggressive / conservative on 4th</b><p>How often the head coach goes for it on 4th down compared with an average coach in the same spot (distance, field position, score, time left), from every decision he has made since 2006, recent seasons weighted most. Tagged at 3 or more percentage points above or below average. Shown for information: aggressive coaches' teams do score a few more touchdowns than their betting line implies, but the betting markets and each player's own scoring history already account for it, so it doesn't change the model's numbers.</p></div>
      <div><b>Projection</b><p>The model's number blended with the betting market. History shows the market is usually closer, so it gets most of the weight.</p></div>
      <div><b>Edge (EV)</b><p>Expected profit per dollar bet. +4% means $4 expected profit per $100 over many bets, not on this one.</p></div>
      <div><b>Fair price</b><p>The moneyline that would make a bet break even at the model's win chance. Compare it with your book.</p></div>
      <div><b>Other books</b><p>On a prop, the chance the other sportsbooks (FanDuel, BetMGM, Caesars and others) give our side at DraftKings' line, from the median of their no-vig prices. Green means they also see value at DraftKings' price, red means they don't. Shown for information while it's tested; it doesn't change the model's numbers yet.</p></div>
      <div><b>Lean / No edge</b><p>A lean is a small positive edge below the betting bar. No edge means both sides are worse than the price.</p></div>
    </div></section>
</main>
<footer><div class="wrap">Generated {generated} from nflverse data through {dt}. Lines move all week, so rerun before betting.
For entertainment and analysis. Bet only what you can afford to lose. Help is available at 1-800-GAMBLER.</div></footer>
<script>{TABS_JS}{UI_JS}</script>
</body></html>"""


# ------------------------------------------------------------------ bets right now, filters, results (Oct 27, 2026)
STALE_MIN = 180          # a price older than this (minutes) is flagged as stale on the page


def _asof_iso(s, now) -> str:
    """'Tue Oct 27, 1:00 PM CT' -> ISO timestamp with the Central offset ('' if it can't be read)."""
    from zoneinfo import ZoneInfo
    try:
        t = str(s).replace(" CT", "").strip()
        m = __import__("re").search(r"[A-Z][a-z]{2} [A-Z][a-z]{2} \d{1,2}, \d{1,2}:\d{2} [AP]M", t)
        if not m:
            return ""
        d = datetime.strptime(f"{now.year} {m.group(0)}", "%Y %a %b %d, %I:%M %p").replace(tzinfo=ZoneInfo("America/Chicago"))
        if d > now + timedelta(days=2):
            d = d.replace(year=now.year - 1)
        return d.isoformat()
    except Exception:
        return ""


def _mkt_key(market: str, stat: str = "") -> str:
    if market == "prop":
        return "td" if stat == "anytime_td" else "prop"
    return {"spread": "side", "total": "total", "moneyline": "ml"}.get(market, market)


def _fx_attrs(book: str, mk: str, ev: float) -> str:
    kind = "prop" if mk in ("prop", "td") else "game"
    return f'data-book="{"fd" if book == "FanDuel" else "dk"}" data-mk="{mk}" data-kind="{kind}" data-ev="{ev:.4f}"'


def _kind_toggle(n_game: int, n_prop: int) -> str:
    """Game bets / Player props switch at the top of Best bets (Oct 28): remembered on the device."""
    b = lambda v, lab, n: f'<button data-f="kind" data-v="{v}" aria-pressed="{str(v == "all").lower()}">{lab} <span class="cnt">{n}</span></button>'
    return (f'<div class="seg fbar" role="group" aria-label="Show game bets or player props">'
            f'{b("all", "All", n_game + n_prop)}{b("game", "Game bets", n_game)}{b("prop", "Player props", n_prop)}</div>')


FILTER_BAR = """<div class="fbar" role="group" aria-label="Filter bets">
  <div class="fgrp"><span>Book</span><button data-f="book" data-v="all" aria-pressed="true">Both</button><button data-f="book" data-v="dk" aria-pressed="false">DraftKings</button><button data-f="book" data-v="fd" aria-pressed="false">FanDuel</button></div>
  <div class="fgrp"><span>Min edge</span><button data-f="ev" data-v="0" aria-pressed="true">Any</button><button data-f="ev" data-v="0.05" aria-pressed="false">5%+</button><button data-f="ev" data-v="0.08" aria-pressed="false">8%+</button></div>
</div>"""


SHOW_WHY = True    # "Why this bet" panels on Best bets (approved by Alex Oct 28, 2026)


def _why(make) -> str:
    """'Why this bet' panel: a tap-to-open list of the reasons (empty if they can't be built)."""
    if not SHOW_WHY:
        return ""
    try:
        lines = make()
    except Exception as e:                      # an explanation must never break the page
        print(f"  why skipped ({e})")
        return ""
    if not lines:
        return ""
    items = "".join((f"<li><b>{escape(t[0])}.</b> {escape(t[1])}</li>" if isinstance(t, tuple) else f"<li>{escape(t)}</li>")
                    for t in lines)
    return f'<details class="why"><summary>Why this bet</summary><ul>{items}</ul></details>'


def _why_ctx(ratings, play, injuries, market_info) -> dict:
    if not SHOW_WHY:
        return {}
    try:
        return W.team_context(ratings, play, injuries, (market_info or {}).get("qb"), (market_info or {}).get("qb_status"))
    except Exception as e:
        print(f"  why context skipped ({e})")
        return {}


def _now_section(upcoming, props, log, dk_used, now, sm=None, wctx=None) -> str:
    """Every bet that clears the bar right now, one list: pick, book, price, edge, kickoff and how old the price is.
    Below it, bets logged earlier whose price has since moved so they no longer clear the bar."""
    rows = []
    for it in upcoming:
        g = it["g"]
        key = f"{g.away_team} @ {g.home_team}"
        asof = dk_used.get(key, "")
        for o in it["opts"]:
            if o["ev"] >= C.MIN_EV:
                rows.append(dict(ev=o["ev"], pick=o["pick"].replace(" (FanDuel)", ""), game=key, when=_kickoff(g),
                                 book=o.get("book", "DraftKings"), odds=o["odds"], win=o["win_prob"], asof=asof,
                                 mk=_mkt_key(o["market"]), extra="", why=_why(lambda: W.game_why(g, o, sm, wctx) if sm is not None else [])))
    if props is not None and not props.empty and "flag" in props:
        for r in props[props.flag.astype(bool) & props.line.notna()].itertuples():
            what = "Anytime TD" if r.stat == "anytime_td" else f"{r.best_side} {r.line:g} {STAT_LABEL.get(r.stat, r.stat).lower()}"
            price = r.over_odds if r.best_side == "Over" else r.under_odds
            win = r.p_over if r.best_side == "Over" else 1 - r.p_over
            rows.append(dict(ev=r.ev, pick=f"{r.player} {what}", game=r.game, when="", book=str(getattr(r, "book", "") or "DraftKings"),
                             odds=price, win=win, asof=str(r.as_of), mk=_mkt_key("prop", r.stat), extra=_books_note(r),
                             why=_why(lambda: W.prop_why(r))))
    rows.sort(key=lambda x: -x["ev"])
    game_rows = [x for x in rows if x["mk"] not in ("prop", "td")]
    prop_rows = [x for x in rows if x["mk"] in ("prop", "td")]
    toggle = _kind_toggle(len(game_rows), len(prop_rows))

    def _rows_html(rows):
        return "".join(
            f"""<tr class="fx" {_fx_attrs(x['book'], x['mk'], x['ev'])}><td><b>{escape(x['pick'])}</b>
        <span class="cell-sub">{escape(x['game'])}{(' · ' + x['when']) if x['when'] else ''}{x['extra']}</span></td>
        <td><span class="bk bk-{'fd' if x['book'] == 'FanDuel' else 'dk'}">{'FD' if x['book'] == 'FanDuel' else 'DK'}</span></td>
        <td class="n">{_odds(x['odds'])}<span class="cell-sub age" data-asof="{_asof_iso(x['asof'], now)}">{escape(str(x['asof']).replace(' CT', ''))}</span></td>
        <td class="n hide-sm">{_pct(x['win'], 1)}</td><td class="n good">{_signed_pct(x['ev'])}</td></tr>"""
            + (f"""<tr class="fx why-row" {_fx_attrs(x['book'], x['mk'], x['ev'])}><td colspan="5">{x['why']}</td></tr>""" if x['why'] else "")
            for x in rows)

    def _group(kind, title, rows, empty):
        if not rows:
            return f'<div class="betgroup" data-kind="{kind}"><h3>{title}</h3><p class="empty">{empty}</p></div>'
        return (f'<div class="betgroup" data-kind="{kind}"><h3>{title} <span class="cnt">{len(rows)}</span></h3>'
                f'<div class="tablebox"><table class="nowtable"><thead><tr><th>Bet</th><th>Book</th><th class="n">Price</th>'
                f'<th class="n hide-sm">Win %</th><th class="n">Edge</th></tr></thead><tbody>{_rows_html(rows)}</tbody></table></div>'
                f'<p class="empty gnone" hidden>No {title.lower()} match these filters.</p></div>')
    if rows:
        table = (_group("game", "Game bets", game_rows, "No spread, total or moneyline clears the bar right now.")
                 + _group("prop", "Player props", prop_rows, "No player prop clears the bar right now."))
    else:
        table = f'<p class="empty">No bet clears the {C.MIN_EV:.0%} bar right now. Leans are on the Leans tab.</p>'

    # bets logged earlier that no longer clear the bar at the current price
    moved = []
    if log is not None and len(log) and "tier" in log:
        cur_game = {}
        for it in upcoming:
            for o in it["opts"]:
                cur_game[(it["g"].game_id, o["market"], o["side"])] = o
        cur_prop = {}
        if props is not None and not props.empty:
            for r in props[props.line.notna()].itertuples():
                cur_prop[(r.game, r.player, r.stat)] = r
        live_ids = {it["g"].game_id for it in upcoming}
        lg = log[(log.tier.fillna("bet") == "bet") & log.game_id.isin(live_ids)]
        for r in lg.itertuples():
            if r.market == "prop":
                c = cur_prop.get((r.game, r.player, r.stat))
                if c is None:
                    continue
                same_side = c.best_side == r.side
                if same_side and bool(c.flag):
                    continue
                now_price = c.over_odds if r.side == "Over" else c.under_odds
                now_ev = c.ev if same_side else np.nan
                from .props import LEAN_ONLY_STATS
                why = ("leans only for now" if same_side and c.stat in LEAN_ONLY_STATS and c.ev >= C.MIN_EV
                       else "model now prefers the other side" if not same_side
                       else "price moved" if pd.isna(r.odds) or pd.isna(now_price) or abs(float(now_price) - float(r.odds)) > 0.5
                       else "model updated (same price)")
            else:
                c = cur_game.get((r.game_id, r.market, r.side))
                if c is None or c["ev"] >= C.MIN_EV:
                    continue
                now_price, now_ev = c["odds"], c["ev"]
                why = "price moved" if pd.isna(r.odds) or abs(float(now_price) - float(r.odds)) > 0.5 else "model updated (same price)"
            moved.append(f"""<tr><td>{escape(_pretty_pick(r.pick))}<span class="cell-sub">{escape(str(r.game))} · flagged {escape(str(r.logged_at))}</span></td>
        <td class="n">{_odds(r.odds)}<span class="cell-sub">{_signed_pct(r.ev)}</span></td>
        <td class="n">{_odds(now_price)}<span class="cell-sub">{_signed_pct(now_ev) if pd.notna(now_ev) else '—'}</span></td>
        <td class="muted">{why}</td></tr>""")
    moved_html = ""
    if moved:
        moved_html = f"""<h3>No longer a bet at today's price</h3>
    <p class="note">Flagged earlier, but the line or price has moved and the edge is now under {C.MIN_EV:.0%}. If you already have it at the old price, it's still tracked at that price.</p>
    <div class="tablebox"><table><thead><tr><th>Bet</th><th class="n">Then</th><th class="n">Now</th><th>Why</th></tr></thead>
    <tbody>{''.join(moved)}</tbody></table></div>"""
    return f"""<section id="bets"><h2>Best bets right now</h2>
    <p class="lede">Every bet that clears the {C.MIN_EV:.0%} edge bar at the latest DraftKings or FanDuel price, split into game bets and player props, best edge first. Use the switch to show one or both.
    Each price shows how long ago it was checked; one older than {STALE_MIN // 60} hours is marked <span class="stale-key">stale</span>, so confirm it in your app.</p>
    {toggle if rows else ''}
    {FILTER_BAR if rows else ''}
    {table}
    {moved_html}"""


def _results_section(log) -> str:
    """Live results by week, market and book, plus cumulative units over time."""
    if log is None or not len(log):
        return '<section id="results"><h2>Results</h2><p class="empty">No results yet.</p></section>'
    lg = log.copy()
    lg["tier"] = lg["tier"].fillna("bet") if "tier" in lg else "bet"
    if "book" not in lg:
        lg["book"] = np.nan
    lg["book"] = lg.book.where(lg.book.notna() & (lg.book.astype(str) != ""),
                               np.where(lg.pick.astype(str).str.contains("(FanDuel)", regex=False), "FanDuel", "DraftKings"))
    lg["kind"] = np.where(lg.market == "prop", "Prop: " + lg.stat.map(lambda s: STAT_LABEL.get(s, str(s))).astype(str),
                          lg.market.map({"spread": "Spread", "total": "Total", "moneyline": "Moneyline"}).fillna(lg.market))
    st = lg[lg.result.isin(["win", "loss", "push"])].copy()
    st["pnl"] = st.pnl.astype(float)
    closed = lg[lg.clv.notna() & (pd.to_datetime(lg.kickoff, utc=True, errors="coerce") <= pd.Timestamp.now(tz="UTC"))]

    def agg(df_settled, df_closed, by):
        keys = sorted(set(df_settled[by]) | set(df_closed[by]), key=str)
        out = []
        for k in keys:
            s, c = df_settled[df_settled[by] == k], df_closed[df_closed[by] == k]
            w, l_ = int((s.result == "win").sum()), int((s.result == "loss").sum())
            p = int((s.result == "push").sum())
            out.append(dict(k=k, rec=f"{w}-{l_}" + (f"-{p}" if p else ""), units=float(s.pnl.sum()) if len(s) else 0.0,
                            roi=float(s.pnl.mean()) if len(s) else np.nan, clv=float(c.clv.mean()) if len(c) else np.nan,
                            beat=float((c.clv > 0).mean()) if len(c) else np.nan, n=len(s)))
        return out

    def table(rows, label):
        if not rows:
            return '<p class="empty">Nothing settled yet.</p>'
        trs = "".join(
            f"""<tr><td><b>{escape(str(r['k']))}</b></td><td class="n">{r['rec']}</td>
            <td class="n {'good' if r['units'] > 0 else 'bad' if r['units'] < 0 else ''}">{r['units']:+.1f}</td>
            <td class="n hide-sm">{_signed_pct(r['roi']) if r['n'] else '—'}</td>
            <td class="n {'good' if pd.notna(r['clv']) and r['clv'] > 0 else 'bad' if pd.notna(r['clv']) and r['clv'] < 0 else ''}">{_signed_pct(r['clv']) if pd.notna(r['clv']) else '—'}</td>
            <td class="n hide-sm">{f"{r['beat']:.0%}" if pd.notna(r['beat']) else '—'}</td></tr>""" for r in rows)
        return f"""<div class="tablebox"><table><thead><tr><th>{label}</th><th class="n">W-L</th><th class="n">Units</th>
        <th class="n hide-sm">ROI</th><th class="n">Avg CLV</th><th class="n hide-sm">Beat close</th></tr></thead><tbody>{trs}</tbody></table></div>"""

    blocks = []
    for tier, title in (("bet", "Bets"), ("lean", "Leans")):
        s, c = st[st.tier == tier], closed[closed.tier == tier]
        s = s.assign(wk="Week " + s.week.astype(int).astype(str))
        c = c.assign(wk="Week " + c.week.astype(int).astype(str))
        wk = sorted(agg(s, c, "wk"), key=lambda r: -int(r["k"].split()[-1]))
        blocks.append(f"""<div><h3>{title} by week</h3>{table(wk, 'Week')}
        <h3>{title} by market</h3>{table(agg(s, c, 'kind'), 'Market')}
        <h3>{title} by book</h3>{table(agg(s, c, 'book'), 'Book')}</div>""")

    # cumulative units, bets and leans, in kickoff order
    chart = ""
    if len(st):
        st["_ko"] = pd.to_datetime(st.kickoff, utc=True, errors="coerce")
        series = {}
        st = st.sort_values(["_ko", "logged_at"]).reset_index(drop=True)
        st["_ord"] = np.arange(1, len(st) + 1)          # one shared order (kickoff, then when logged) for both lines
        for tier in ("bet", "lean"):
            s = st[st.tier == tier]
            if len(s):
                ords = s._ord.to_numpy(float)
                series[tier] = (np.concatenate([[0.0], ords]), np.concatenate([[0.0], s.pnl.cumsum().to_numpy()]))
        if series:
            W, H, P = 520, 230, 38
            t_max = max(v[0].max() for v in series.values()) or 1.0
            lo = min(0.0, min(v[1].min() for v in series.values()))
            hi = max(0.0, max(v[1].max() for v in series.values()))
            span = (hi - lo) or 1.0
            x = lambda t: P + (W - 2 * P) * (t / t_max)
            y = lambda v: 12 + (H - 36) * (hi - v) / span
            lines = []
            for tier, (ts, v) in series.items():
                pts = " ".join(f"{x(t):.1f},{y(val):.1f}" for t, val in zip(ts, v))
                lines.append(f'<polyline class="cu cu-{tier}" points="{pts}"/>'
                             f'<text class="cu-lab cu-{tier}" x="{x(ts[-1]) + 6:.0f}" y="{y(v[-1]) + 4:.1f}">{v[-1]:+.1f}u</text>')
            first = st._ko.min().tz_convert("America/Chicago").strftime("%b %-d")
            last = st._ko.max().tz_convert("America/Chicago").strftime("%b %-d")
            chart = f"""<h3>Units over time</h3>
    <figure class="chart"><svg viewBox="0 0 {W + 50} {H}" role="img" aria-label="Cumulative units won, bets and leans, in kickoff order">
      <line class="zero" x1="{P}" x2="{W - P}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>
      <text class="ax" x="{P - 6}" y="{y(hi) + 4:.1f}" text-anchor="end">{hi:+.0f}</text>
      <text class="ax" x="{P - 6}" y="{y(lo) + 4:.1f}" text-anchor="end">{lo:+.0f}</text>
      <text class="ax" x="{P - 6}" y="{y(0) + 4:.1f}" text-anchor="end">0</text>
      {''.join(lines)}
      <text class="ax" x="{P}" y="{H - 6}">first</text><text class="ax" x="{W - P}" y="{H - 6}" text-anchor="end">latest</text>
    </svg><figcaption><span class="key key-bet"></span>Bets <span class="key key-lean"></span>Leans · one unit staked per bet, in kickoff order ({first} to {last})</figcaption></figure>"""

    tot = {t: st[st.tier == t] for t in ("bet", "lean")}
    cl = {t: closed[closed.tier == t] for t in ("bet", "lean")}
    def tile(t, lab):
        s, c = tot[t], cl[t]
        w, l_ = int((s.result == "win").sum()), int((s.result == "loss").sum())
        clv = _signed_pct(c.clv.mean()) if len(c) else "—"
        return (f'<div><span>{lab}</span><b>{w}-{l_}</b><em>{s.pnl.sum():+.1f} units · avg CLV '
                f'<i class="{"good" if len(c) and c.clv.mean() > 0 else "bad" if len(c) and c.clv.mean() < 0 else ""}">{clv}</i> · {len(c)} closed</em></div>')
    return f"""<section id="results"><h2>Results</h2>
    <p class="lede">Every flagged bet and every lean since tracking began (Week 7, 2026), graded at the price it was posted.
    CLV compares that price with the closing line: over hundreds of bets it's the most reliable sign of a real edge, well before win-loss settles down.
    Bets placed at FanDuel are measured against DraftKings' closing line, so their CLV includes the better price.</p>
    <div class="leanstrip">{tile('bet', 'Bets')}{tile('lean', 'Leans')}</div>
    {chart}
    <div class="best">{''.join(blocks)}</div></section>"""


UI_JS = """
(function(){
  // price age: "12 min ago", stale after STALE minutes
  function ages(){
    var now=Date.now();
    [].forEach.call(document.querySelectorAll('[data-asof]'),function(el){
      var t=Date.parse(el.dataset.asof); if(!el.dataset.asof||isNaN(t)) return;
      var m=Math.max(0,Math.round((now-t)/60000));
      var txt=m<60?(m+' min ago'):(m<1440?(Math.floor(m/60)+' h '+(m%60)+' min ago'):(Math.floor(m/1440)+' d ago'));
      el.textContent=txt; el.classList.toggle('stale', m>STALE); if(m>STALE) el.title='Price checked '+txt+': confirm it in your app';
    });
  }
  ages(); setInterval(ages, 60000);
  // filters for the bets list
  var state={book:'all',kind:'all',ev:'0'};
  try{var s=JSON.parse(localStorage.getItem('rr_filters')||'null'); if(s){delete s.mk; state=Object.assign(state,s);}}catch(e){}
  function apply(){
    var rows=[].slice.call(document.querySelectorAll('tr.fx')), shown=0;
    rows.forEach(function(r){
      var ok=(state.book==='all'||r.dataset.book===state.book)&&(parseFloat(r.dataset.ev)>=parseFloat(state.ev));
      r.hidden=!ok; if(ok) shown++;
    });
    [].forEach.call(document.querySelectorAll('.betgroup'),function(g){
      g.hidden=!(state.kind==='all'||g.dataset.kind===state.kind);
      var rs=g.querySelectorAll('tr.fx'), vis=0; [].forEach.call(rs,function(r){if(!r.hidden) vis++;});
      var gn=g.querySelector('.gnone'); if(gn) gn.hidden=!(rs.length&&!vis);
      var tb=g.querySelector('.tablebox'); if(tb) tb.hidden=rs.length&&!vis;
    });
    [].forEach.call(document.querySelectorAll('.fbar button'),function(b){b.setAttribute('aria-pressed', String(state[b.dataset.f]===b.dataset.v))});
  }
  [].forEach.call(document.querySelectorAll('.fbar button'),function(b){
    b.addEventListener('click',function(){state[b.dataset.f]=b.dataset.v; try{localStorage.setItem('rr_filters',JSON.stringify(state))}catch(e){} apply();});
  });
  apply();
  var phone=window.matchMedia&&window.matchMedia('(max-width:640px)').matches;
  // phones: touchdown and prop panels start collapsed on each game card
  if(phone){
    [].forEach.call(document.querySelectorAll('details.xtra'),function(d){d.open=false});
  }
  // phones: game cards start compact (teams, lines, best bet); tap to open the rest
  if(phone){
    document.documentElement.classList.add('cardsjs');
    [].forEach.call(document.querySelectorAll('article.card'),function(c){
      var b=c.querySelector('.card-more'); if(!b) return;
      c.classList.add('collapsed');
      function set(open){c.classList.toggle('collapsed',!open); b.setAttribute('aria-expanded',String(open));
        b.textContent=open?'Hide details':'Lines, props and news';}
      b.addEventListener('click',function(){var open=c.classList.contains('collapsed'); set(open);
        if(!open){var r=c.getBoundingClientRect(); if(r.top<0) window.scrollBy(0,r.top-56);}});
    });
    // long news notes: 3 lines with a "more" link
    [].forEach.call(document.querySelectorAll('.card-news p'),function(p){
      if(p.textContent.length<190) return;
      p.classList.add('clamp');
      var m=document.createElement('button'); m.type='button'; m.className='news-more'; m.textContent='More';
      m.addEventListener('click',function(){var c=p.classList.toggle('clamp'); m.textContent=c?'More':'Less';});
      p.parentNode.appendChild(m);
    });
  }
  // tab bar: fade the right edge while more tabs are hidden off-screen
  var bar=document.querySelector('nav.toc .wrap');
  if(bar){var edge=function(){bar.classList.toggle('fade-r', bar.scrollLeft+bar.clientWidth<bar.scrollWidth-4);
      bar.classList.toggle('fade-l', bar.scrollLeft>4);};
    bar.addEventListener('scroll',edge,{passive:true}); window.addEventListener('resize',edge); edge();}
})();
""".replace("STALE", str(STALE_MIN))

UI_CSS = """
.fbar{display:flex;flex-wrap:wrap;gap:8px 18px;margin:0 0 12px}
.fgrp{display:flex;flex-wrap:wrap;align-items:center;gap:4px}
.fgrp>span{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-right:4px}
.fbar button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:999px;padding:4px 11px;cursor:pointer;min-height:32px}
.fbar button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.fbar button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.bk{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.04em;border-radius:3px;padding:1px 6px;border:1px solid var(--line)}
.bk-fd{color:var(--accent);border-color:var(--accent)}
.age.stale,.stale-key{color:var(--flag);font-weight:600}
.nowtable td{vertical-align:middle}
@media (max-width:520px){.hide-sm{display:none}}
.chart{margin:0 0 18px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.chart svg{width:100%;height:auto;display:block}
.chart .zero{stroke:var(--line);stroke-width:1}
.chart .ax{font-size:13px;fill:var(--muted);font-family:var(--mono)}
.chart .cu{fill:none;stroke-width:2.5;stroke-linejoin:round}
.chart .cu-bet{stroke:var(--accent)}.chart .cu-lean{stroke:var(--muted);stroke-dasharray:5 4}
.chart .cu-lab{font-size:14px;font-weight:600;font-family:var(--mono)}.chart text.cu-bet{fill:var(--accent)}.chart text.cu-lean{fill:var(--muted)}
.chart figcaption{font-size:12.5px;color:var(--muted);margin-top:4px}
#results .best h3{margin-top:22px}#results .best h3:first-child{margin-top:0}
.key{display:inline-block;width:16px;height:0;border-top:2.5px solid var(--accent);vertical-align:middle;margin:0 4px 0 8px}
.key-lean{border-top:2.5px dashed var(--muted)}
details.xtra>summary{display:none}
@media (max-width:640px){details.xtra>summary{display:block;cursor:pointer;color:var(--accent);font-weight:600;font-size:14px;padding:10px 14px;border-top:1px solid var(--line)}
 details.xtra>summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}}
"""


# Logo (Oct 27, 2026): original vector badge, a rat in sunglasses and a green cap. The full badge with "THE RAT REPORT"
# on the cap is assets/logo-badge.svg; the header and browser tab use the same art without the cap text, which is
# unreadable that small (the title sits right beside it).
LOGO_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" role="img" aria-label="The Rat Report logo: a rat in sunglasses and a green cap">\n  <defs>\n    <radialGradient id="rrbg" cx="50%" cy="38%" r="70%"><stop offset="0" stop-color="#22372B"/><stop offset="1" stop-color="#14211A"/></radialGradient>\n    <linearGradient id="rrcap" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1E7A4B"/><stop offset="1" stop-color="#155C38"/></linearGradient>\n    <linearGradient id="rrlens" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2A3530"/><stop offset=".55" stop-color="#0B0F0D"/></linearGradient>\n    <clipPath id="rrclip"><circle cx="100" cy="100" r="92"/></clipPath>\n  </defs>\n  <circle cx="100" cy="100" r="98" fill="#C99700"/>\n  <circle cx="100" cy="100" r="93" fill="url(#rrbg)"/>\n  <circle cx="100" cy="100" r="88" fill="none" stroke="#C99700" stroke-width="1.2" opacity=".55"/>\n  <g clip-path="url(#rrclip)" transform="translate(0,8)">\n    <!-- ears -->\n    <circle cx="57" cy="84" r="25" fill="#8E9A94"/><circle cx="57" cy="85" r="14.5" fill="#E9A7AE"/>\n    <circle cx="143" cy="84" r="25" fill="#8E9A94"/><circle cx="143" cy="85" r="14.5" fill="#E9A7AE"/>\n    <!-- head -->\n    <path d="M100 64 C139 64 151 95 147 118 C142 143 118 163 100 171 C82 163 58 143 53 118 C49 95 61 64 100 64 Z" fill="#A9B4AE"/>\n    <path d="M100 64 C139 64 151 95 147 118 C146 124 144 129 141 134 C142 108 128 82 100 80 C72 82 58 108 59 134 C56 129 54 124 53 118 C49 95 61 64 100 64 Z" fill="#98A49E"/>\n    <!-- muzzle -->\n    <path d="M100 127 C117 127 125 140 121 152 C117 163 107 168 100 169 C93 168 83 163 79 152 C75 140 83 127 100 127 Z" fill="#CBD3CE"/>\n    <ellipse cx="100" cy="151" rx="7.5" ry="5.2" fill="#E9A7AE"/>\n    <ellipse cx="98" cy="149.5" rx="2.2" ry="1.3" fill="#fff" opacity=".6"/>\n    <path d="M100 156.5 V161.5 M94 163.5 Q100 168 106 163.5" stroke="#5C6C63" stroke-width="2" fill="none" stroke-linecap="round"/>\n    <!-- whiskers -->\n    <g stroke="#DDE4E0" stroke-width="1.5" stroke-linecap="round" opacity=".8" fill="none">\n      <path d="M83 149 Q66 144 48 143"/><path d="M83 154 Q65 155 47 160"/><path d="M117 149 Q134 144 152 143"/><path d="M117 154 Q135 155 153 160"/>\n    </g>\n    <!-- sunglasses -->\n    <path d="M60 113 Q100 107 140 113" stroke="#0B0F0D" stroke-width="4.5" fill="none" stroke-linecap="round"/>\n    <path d="M63 111 H97 V119 Q97 131 84 131 H76 Q63 131 63 119 Z" fill="url(#rrlens)"/>\n    <path d="M103 111 H137 V119 Q137 131 124 131 H116 Q103 131 103 119 Z" fill="url(#rrlens)"/>\n    <path d="M69 116 L78 116" stroke="#fff" stroke-width="2.4" stroke-linecap="round" opacity=".7"/>\n    <path d="M109 116 L118 116" stroke="#fff" stroke-width="2.4" stroke-linecap="round" opacity=".7"/>\n    <!-- cap -->\n    <path d="M45 98 C45 45 155 45 155 98 Z" fill="url(#rrcap)"/>\n    <path d="M100 58 C96 70 96 84 100 98" stroke="#0F4A2D" stroke-width="1.6" fill="none" opacity=".7"/>\n    <circle cx="100" cy="57.5" r="4.5" fill="#0F4A2D"/>\n    <path d="M37 98 Q100 86 163 98 Q100 118 37 98 Z" fill="#0F4A2D"/>\n    <path d="M37 98 Q100 86 163 98" stroke="#1E7A4B" stroke-width="1.2" fill="none"/>\n    \n  </g>\n</svg>'
FAVICON = "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'%3E %3Cdefs%3E %3CradialGradient id='rrbg' cx='50%25' cy='38%25' r='70%25'%3E%3Cstop offset='0' stop-color='%2322372B'/%3E%3Cstop offset='1' stop-color='%2314211A'/%3E%3C/radialGradient%3E %3ClinearGradient id='rrcap' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0' stop-color='%231E7A4B'/%3E%3Cstop offset='1' stop-color='%23155C38'/%3E%3C/linearGradient%3E %3ClinearGradient id='rrlens' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0' stop-color='%232A3530'/%3E%3Cstop offset='.55' stop-color='%230B0F0D'/%3E%3C/linearGradient%3E %3CclipPath id='rrclip'%3E%3Ccircle cx='100' cy='100' r='92'/%3E%3C/clipPath%3E %3C/defs%3E %3Ccircle cx='100' cy='100' r='98' fill='%23C99700'/%3E %3Ccircle cx='100' cy='100' r='93' fill='url%28%23rrbg%29'/%3E %3Ccircle cx='100' cy='100' r='88' fill='none' stroke='%23C99700' stroke-width='1.2' opacity='.55'/%3E %3Cg clip-path='url%28%23rrclip%29' transform='translate%280%2C8%29'%3E %3C%21-- ears --%3E %3Ccircle cx='57' cy='84' r='25' fill='%238E9A94'/%3E%3Ccircle cx='57' cy='85' r='14.5' fill='%23E9A7AE'/%3E %3Ccircle cx='143' cy='84' r='25' fill='%238E9A94'/%3E%3Ccircle cx='143' cy='85' r='14.5' fill='%23E9A7AE'/%3E %3C%21-- head --%3E %3Cpath d='M100 64 C139 64 151 95 147 118 C142 143 118 163 100 171 C82 163 58 143 53 118 C49 95 61 64 100 64 Z' fill='%23A9B4AE'/%3E %3Cpath d='M100 64 C139 64 151 95 147 118 C146 124 144 129 141 134 C142 108 128 82 100 80 C72 82 58 108 59 134 C56 129 54 124 53 118 C49 95 61 64 100 64 Z' fill='%2398A49E'/%3E %3C%21-- muzzle --%3E %3Cpath d='M100 127 C117 127 125 140 121 152 C117 163 107 168 100 169 C93 168 83 163 79 152 C75 140 83 127 100 127 Z' fill='%23CBD3CE'/%3E %3Cellipse cx='100' cy='151' rx='7.5' ry='5.2' fill='%23E9A7AE'/%3E %3Cellipse cx='98' cy='149.5' rx='2.2' ry='1.3' fill='%23fff' opacity='.6'/%3E %3Cpath d='M100 156.5 V161.5 M94 163.5 Q100 168 106 163.5' stroke='%235C6C63' stroke-width='2' fill='none' stroke-linecap='round'/%3E %3C%21-- whiskers --%3E %3Cg stroke='%23DDE4E0' stroke-width='1.5' stroke-linecap='round' opacity='.8' fill='none'%3E %3Cpath d='M83 149 Q66 144 48 143'/%3E%3Cpath d='M83 154 Q65 155 47 160'/%3E%3Cpath d='M117 149 Q134 144 152 143'/%3E%3Cpath d='M117 154 Q135 155 153 160'/%3E %3C/g%3E %3C%21-- sunglasses --%3E %3Cpath d='M60 113 Q100 107 140 113' stroke='%230B0F0D' stroke-width='4.5' fill='none' stroke-linecap='round'/%3E %3Cpath d='M63 111 H97 V119 Q97 131 84 131 H76 Q63 131 63 119 Z' fill='url%28%23rrlens%29'/%3E %3Cpath d='M103 111 H137 V119 Q137 131 124 131 H116 Q103 131 103 119 Z' fill='url%28%23rrlens%29'/%3E %3Cpath d='M69 116 L78 116' stroke='%23fff' stroke-width='2.4' stroke-linecap='round' opacity='.7'/%3E %3Cpath d='M109 116 L118 116' stroke='%23fff' stroke-width='2.4' stroke-linecap='round' opacity='.7'/%3E %3C%21-- cap --%3E %3Cpath d='M45 98 C45 45 155 45 155 98 Z' fill='url%28%23rrcap%29'/%3E %3Cpath d='M100 58 C96 70 96 84 100 98' stroke='%230F4A2D' stroke-width='1.6' fill='none' opacity='.7'/%3E %3Ccircle cx='100' cy='57.5' r='4.5' fill='%230F4A2D'/%3E %3Cpath d='M37 98 Q100 86 163 98 Q100 118 37 98 Z' fill='%230F4A2D'/%3E %3Cpath d='M37 98 Q100 86 163 98' stroke='%231E7A4B' stroke-width='1.2' fill='none'/%3E %3C/g%3E %3C/svg%3E"
UI_CSS += """
.brand{display:flex;align-items:center;gap:14px;min-width:0}
.brand .logo{flex:none;width:68px;height:68px;display:block}
.brand .logo svg{width:100%;height:100%;display:block;filter:drop-shadow(0 1px 2px rgba(0,0,0,.35))}
@media (max-width:640px){.brand{gap:10px}.brand .logo{width:48px;height:48px}}
"""

# Header logo: Alex's rat-in-a-manhole art (assets/logo-rat.webp), embedded so the
# page stays one file. Falls back to the vector badge if the image is missing.
import base64 as _b64, pathlib as _pl
_ASSETS = _pl.Path(__file__).resolve().parent.parent / "assets"
def _data_uri(name, mime):
    try:
        return f"data:{mime};base64," + _b64.b64encode((_ASSETS / name).read_bytes()).decode()
    except OSError:
        return None
_logo_uri = _data_uri("logo-rat.webp", "image/webp")
_fav_uri = _data_uri("favicon-rat.png", "image/png")
LOGO_HTML = (f'<img src="{_logo_uri}" width="400" height="337" alt="The Rat Report logo">'
             if _logo_uri else LOGO_SVG)
FAVICON_TAG = (f'<link rel="icon" type="image/png" href="{_fav_uri}">' if _fav_uri
               else f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,{FAVICON}">')
UI_CSS += """
.brand .logo{width:auto;height:auto}
.brand .logo img{display:block;width:118px;height:auto;filter:drop-shadow(0 2px 4px rgba(0,0,0,.45))}
@media (max-width:640px){.brand .logo{width:auto;height:auto}.brand .logo img{width:78px}}
"""

# Header without the big headline (Oct 27): the logo carries the name.
UI_CSS += """
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.mast .wrap{align-items:center;padding-block:18px 18px}
.brand{gap:18px}
.brand .logo img{width:176px}
.brand-meta{display:flex;flex-direction:column;align-items:flex-start;gap:10px}
.brand-meta .kicker{margin:0;opacity:.8;font-size:13px}
.brand-meta .beta{margin-left:0;top:0}
@media (max-width:640px){
 .mast .wrap{padding-block:14px 16px;gap:14px}
 .brand{gap:14px}.brand .logo img{width:128px}
 .brand-meta{gap:8px}.brand-meta .kicker{font-size:11px;line-height:1.5}
 .mast-stats{gap:20px}.mast-stats b{font-size:26px}
}
"""

# Phone layout (Oct 27): compact game cards, stacked prop rows (no sideways scrolling inside the page),
# tab bar fade, clamped news, lazy rendering of off-screen cards, past leans collapsed.
UI_CSS += """
.card-body{display:flex;flex-direction:column;flex:1}
.card-sum,.card-more{display:none}
.card{content-visibility:auto;contain-intrinsic-size:auto 700px}
nav.toc .wrap.fade-r{-webkit-mask-image:linear-gradient(90deg,#000 82%,transparent);mask-image:linear-gradient(90deg,#000 82%,transparent)}
nav.toc .wrap.fade-l.fade-r{-webkit-mask-image:linear-gradient(90deg,transparent,#000 12%,#000 82%,transparent);mask-image:linear-gradient(90deg,transparent,#000 12%,#000 82%,transparent)}
nav.toc .wrap.fade-l:not(.fade-r){-webkit-mask-image:linear-gradient(90deg,transparent,#000 12%);mask-image:linear-gradient(90deg,transparent,#000 12%)}
details.past-leans{margin-top:18px}
details.past-leans>summary{font-weight:600}
@media (max-width:640px){
 nav.toc a{font-size:12px;padding:12px 9px 10px;letter-spacing:.04em}
 .cardsjs .card-sum{display:flex;flex-wrap:wrap;gap:4px 12px;padding:10px 16px 0;font-size:13px;color:var(--muted)}
 .cardsjs .card.collapsed .card-sum{display:flex}
 .cardsjs .card:not(.collapsed) .card-sum{display:none}
 .card-sum .sum-pick{flex-basis:100%;color:var(--ink);font-weight:600}
 .card-sum .sum-pick.bet::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--flag);margin-right:6px;vertical-align:1px}
 .cardsjs .card-more{display:block;width:100%;text-align:left;font:inherit;font-size:14px;font-weight:600;color:var(--accent);background:none;border:0;
   border-top:1px solid var(--line);margin-top:10px;padding:11px 16px;cursor:pointer}
 .cardsjs .card-more::after{content:" ▾"}
 .cardsjs .card:not(.collapsed) .card-more::after{content:" ▴"}
 .cardsjs .card.collapsed .card-body{display:none}
 .cardsjs .card:not(.collapsed) .card-more{margin-top:0;border-bottom:1px solid var(--line);border-top:1px solid var(--line)}
 .card{contain-intrinsic-size:auto 260px}
 .card-news p.clamp{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
 .card-news .news-more{font:inherit;font-size:13px;font-weight:600;color:var(--accent);background:none;border:0;padding:0 0 0 4px;cursor:pointer;align-self:flex-end}
 .card-news{flex-wrap:wrap}
 .card table.props thead{display:none}
 .card table.props,.card table.props tbody{display:block}
 .card table.props tr{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:2px 10px;padding:10px 12px;border-bottom:1px solid var(--line)}
 .card table.props td{display:block;border:0;padding:0;text-align:left}
 .card table.props td:first-child{grid-column:1/-1;margin-bottom:4px}
 .card table.props td[data-l]::before{content:attr(data-l);display:block;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-family:var(--body)}
 .card .tablebox{overflow-x:visible}
 .card table.props td[data-l]:empty{display:none}
}
"""

# Dark mode (Oct 28): follows the phone/computer setting; the moon/sun button in the header switches it and the
# choice is remembered on that device. The header stays dark green in both themes.
UI_CSS += """
:root{--mast-bg:#14211A;--mast-ink:#F1F4F2}
:root[data-theme="dark"]{--mast-bg:#0A100D}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--mast-bg:#0A100D}}
.mast{background:var(--mast-bg);color:var(--mast-ink);border-bottom:1px solid var(--line)}
.mast-stats{align-items:center}
.theme-btn{flex:none;width:40px;height:40px;border-radius:50%;border:1px solid rgba(241,244,242,.28);background:transparent;color:var(--mast-ink);
  cursor:pointer;display:grid;place-items:center;padding:0;align-self:center}
.theme-btn:hover{border-color:rgba(241,244,242,.6)}
.theme-btn:focus-visible{outline:2px solid var(--flag);outline-offset:2px}
.theme-btn span{width:18px;height:18px;border-radius:50%;box-shadow:inset -6px -3px 0 0 var(--mast-ink);transform:rotate(-20deg)}
:root[data-theme="dark"] .theme-btn span{box-shadow:none;background:var(--mast-ink);width:12px;height:12px;
  outline:3px dotted rgba(241,244,242,.75);outline-offset:3px}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .theme-btn span{box-shadow:none;background:var(--mast-ink);width:12px;height:12px;
  outline:3px dotted rgba(241,244,242,.75);outline-offset:3px}}
"""
UI_JS += """
(function(){
  var b=document.querySelector('.theme-btn'); if(!b) return;
  var root=document.documentElement, mq=window.matchMedia?window.matchMedia('(prefers-color-scheme: dark)'):null;
  function dark(){var t=root.getAttribute('data-theme'); return t?t==='dark':!!(mq&&mq.matches);}
  function label(){b.setAttribute('aria-label', dark()?'Switch to light mode':'Switch to dark mode'); b.title=dark()?'Light mode':'Dark mode';}
  b.addEventListener('click',function(){var t=dark()?'light':'dark'; root.setAttribute('data-theme',t);
    try{localStorage.setItem('rr_theme',t)}catch(e){} label();});
  if(mq&&mq.addEventListener) mq.addEventListener('change',label);
  label();
})();
"""

UI_CSS += """
details.why{margin-top:6px;font-size:13px}
details.why>summary{cursor:pointer;color:var(--accent);font-weight:600;list-style:none;display:inline-flex;align-items:center;gap:4px}
details.why>summary::-webkit-details-marker{display:none}
details.why>summary::after{content:"\\25BE";font-size:11px}
details.why[open]>summary::after{content:"\\25B4"}
details.why ul{margin:6px 0 2px;padding:8px 10px 8px 24px;background:var(--chip);border-radius:6px;color:var(--ink);max-width:62ch}
details.why li{margin:3px 0;line-height:1.45}
"""

UI_CSS += """
table.nowtable tr.fx:has(+ tr.why-row) td{border-bottom:0;padding-bottom:2px}
table.nowtable tr.why-row td{padding-top:0}
table.nowtable tr.why-row details.why{margin-top:0}
"""

UI_CSS += """
.seg{display:inline-flex;gap:0;border:1px solid var(--line);border-radius:999px;padding:3px;background:var(--surface);margin:0 0 12px;flex-wrap:nowrap}
.seg button{border:0;border-radius:999px;background:transparent;color:var(--ink);font:inherit;font-size:14px;font-weight:600;padding:7px 14px;min-height:36px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--bg)}
.seg .cnt,.betgroup h3 .cnt{font-family:var(--mono);font-size:12px;opacity:.75;margin-left:2px}
.betgroup{margin-top:6px}
.betgroup+.betgroup{margin-top:22px}
.betgroup h3{margin:0 0 8px}
@media (max-width:640px){.seg{display:flex;width:100%}.seg button{flex:1;padding:7px 6px;font-size:13px}}
"""
