"""Render the weekly model output as one self-contained, readable HTML page."""
from __future__ import annotations

from datetime import datetime, timedelta
from html import escape

import numpy as np
import pandas as pd

from . import config as C
from .betting import decimal, fair_american, implied

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
        <span class="cell-sub">{r.team} vs {r.opp} · DK {escape(r.as_of)}</span>{'<span class="inj">Large edge: check news and the current line</span>' if r.ev > 0.15 else ''}</td>
        <td class="n">{_odds(r.over_odds if r.best_side == 'Over' else r.under_odds)}</td>
        <td class="n">{_pct(r.p_over if r.best_side == 'Over' else 1 - r.p_over, 1)}</td>
        <td class="n {'good' if r.flag else ''}">{_signed_pct(r.ev)}{'' if r.flag else ' <span class="muted">lean</span>'}</td></tr>"""
        for r in have.itertuples())
    return f"""<div class="tablebox"><table>
      <thead><tr><th>Prop</th><th class="n">Odds</th><th class="n">Win %</th><th class="n">Edge</th></tr></thead>
      <tbody>{rows}</tbody></table></div>"""


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
    if g.home_qb_change:
        tags.append(f"{g.home_team} QB change")
    if g.away_qb_change:
        tags.append(f"{g.away_team} QB change")
    if g.away_west_early:
        tags.append(f"{g.away_team} early body clock")
    mkt = mkt or {}
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
            better = (book != "DraftKings") and (bo != o["odds"] or (line_now is not None and pd.notna(bl)
                                                                     and abs(abs(bl) - abs(line_now)) > 0.01))
            if better:
                ln = "" if market == "moneyline" or pd.isna(bl) else (f"{bl:+g} " if market == "spread" else f"{bl:g} ")
                alt = f'<span class="cell-sub">Best: {escape(book)} {ln}{_odds(bo)}</span>'
        return f'<div class="{cls}">{word} {escape(o["pick"])} <span class="n">{_signed_pct(o["ev"])}</span>{alt}</div>'

    fav_home = g.proj_margin >= 0
    return f"""<article class="card{' has-bet' if flagged else ''}">
      <header class="card-head">
        <div class="teams">
          <div class="team"><span class="abbr">{g.away_team}</span><span class="nm">{NAMES.get(g.away_team, '')}</span>
            <span class="qb">{escape(str(g.away_qb_name or ''))}</span><span class="pts n">{g.proj_away_pts:.1f}</span></div>
          <div class="team"><span class="abbr">{g.home_team}</span><span class="nm">{NAMES.get(g.home_team, '')}</span>
            <span class="qb">{escape(str(g.home_qb_name or ''))}</span><span class="pts n">{g.proj_home_pts:.1f}</span></div>
        </div>
        <div class="when">{_kickoff(g)}</div>
        <div class="tags">{tag_html}</div>
        {_movement_html(g, mkt)}
      </header>
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
      <div class="card-extra">{_card_tds(tds)}{_card_props(props, gi, (mkt or {}).get('best_props', {}))}</div>
      {_card_injuries(g, inj, (mkt or {}).get('qb'))}
      {_card_news(news, (mkt or {}).get('dk_used', {}).get(game_key))}
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
    recent = log.sort_values("logged_at").tail(12).iloc[::-1]
    rec_rows = "".join(
        f"""<tr><td>{escape(str(r.pick))}<span class="cell-sub">{escape(str(r.game))} · logged {escape(str(r.logged_at))}</span></td>
        <td class="n">{_odds(r.odds)}</td><td class="n">{_signed_pct(r.ev)}</td>
        <td class="n {'good' if pd.notna(r.clv) and r.clv > 0 else 'bad' if pd.notna(r.clv) and r.clv < 0 else ''}">{_signed_pct(r.clv) if pd.notna(r.clv) else '—'}</td>
        <td>{escape(str(r.result)) if isinstance(r.result, str) and r.result else 'Open'}</td></tr>"""
        for r in recent.itertuples())
    return f"""<h3>Closing line value (live)</h3>
    <p class="lede">Every flagged bet is logged at the price when it was first flagged, then re-priced at the closing
    DraftKings line. Positive CLV means the bet was worth more than the closing market said: the earliest reliable sign of a real edge.</p>
    <div class="tablebox"><table><thead><tr><th>Bets</th><th class="n">Logged</th><th class="n">Closed</th><th class="n">Avg CLV</th>
    <th class="n">Beat close</th><th class="n">W-L</th><th class="n">Units</th></tr></thead><tbody>{rows}</tbody></table></div>
    <details class="more"><summary>Latest logged bets</summary><div class="tablebox"><table><thead><tr><th>Bet</th><th class="n">Price</th>
    <th class="n">Edge then</th><th class="n">CLV</th><th>Result</th></tr></thead><tbody>{rec_rows}</tbody></table></div></details>"""


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


def _card_tds(tds) -> str:
    """Five most likely anytime touchdown scorers in a game: chance, fair odds and the DraftKings price."""
    if tds is None or len(tds) == 0:
        return ""
    rows = []
    for r in tds.itertuples():
        dk = getattr(r, "over_odds", float("nan"))
        fair = fair_american(r.adj_projection)
        if pd.notna(dk):
            value = (dk / 100 + 1 if dk > 0 else 100 / -dk + 1) * r.adj_projection > 1
            dk_cell = f'<span class="n tdk{" good" if value else ""}">{_odds(dk)}</span>'
        else:
            dk_cell = '<span class="n tdk muted">—</span>'
        inj = f'<span class="inj">{escape(r.injury)}</span>' if r.injury else ""
        rows.append(f"""<li><span class="tdn">{escape(r.player)} <span class="muted">{r.position} · {r.team}</span>{inj}</span>
        <span class="n tdp">{r.adj_projection * 100:.0f}%</span><span class="n tdo">{_odds(fair)}</span>{dk_cell}</li>""")
    return f"""<div class="tds"><h4>Most likely TD scorers</h4><ol class="tdtable">
      <li class="hd"><span>Player</span><span class="n">Chance</span><span class="n">Fair</span><span class="n">DK</span></li>{"".join(rows)}</ol>
      <p class="cell-sub">Green DK price = pays more than fair odds (a value bet).</p></div>"""


def _prop_rows(grp, gi) -> str:
    """Read-only rows: projection, fair line, and the DraftKings line and edge when one was found."""
    rows = []
    for r in grp.itertuples():
        inj = f'<span class="inj">{escape(r.injury)}</span>' if r.injury else ""
        known = pd.notna(r.line)
        fair = (_odds(fair_american(r.a)) if r.stat == "anytime_td" else f"{r.fair_line:g}")
        if known:
            line = "Yes" if r.stat == "anytime_td" else f"{r.line:g}"
            line += f'<span class="cell-sub">DK {escape(r.as_of)}</span>'
            over = _odds(r.over_odds) if pd.notna(r.over_odds) else "—"
            under = _odds(r.under_odds) if pd.notna(r.under_odds) else "—"
            cls = "hit" if r.flag else ("lean" if r.ev > 0 else "no")
            side = "Yes" if r.stat == "anytime_td" else r.best_side
            edge = f'<span class="out {cls}">{side} {_signed_pct(r.ev)}</span>'
        else:
            line, over, under, edge = '<span class="muted">—</span>', "", "", '<span class="muted">No line yet</span>'
        rows.append(f"""<tr>
          <td><b>{escape(r.player)}</b><span class="cell-sub">{r.position} · {r.team} · {STAT_LABEL.get(r.stat, r.stat)}</span>{inj}</td>
          <td class="n">{_fmt_proj(r.stat, r.adj_projection)}</td><td class="n">{fair}</td>
          <td class="n">{line}</td><td class="n">{over}</td><td class="n">{under}</td><td class="n">{edge}</td></tr>""")
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
            if b and b[0] != "DraftKings" and pd.notna(price) and decimal(b[1]) > decimal(price):
                what += f" · Best: {b[0]} {_odds(b[1])}"
            warn = '<span class="inj">Big edge: check the line</span>' if r.ev > 0.15 else ""
            items.append(f"""<li><span class="tdn"><b>{escape(r.player)}</b>{warn}<span class="cell-sub">{escape(what)}</span></span>
            <span class="n tdk">{_odds(price)}</span>
            <span class="n tdp {'good' if r.flag else ('bad' if r.ev <= 0 else '')}">{_signed_pct(r.ev)}</span>
            <span class="tdo">{tag}</span></li>""")
        top = (f'<ol class="tdtable edges"><li class="hd"><span>Bet</span><span class="n">DK</span><span class="n">Edge</span><span></span></li>'
               + "".join(items) + '</ol>')
    else:
        top = '<p class="cell-sub" style="margin:0">No edge · no DraftKings prop lines found yet for this game.</p>'
    return f"""<div class="tds"><h4>Top prop edges</h4>{top}</div>
      <details class="allprops"><summary>All {len(grp)} props</summary>
        <div class="tablebox"><table class="props">
        <thead><tr><th>Player · prop</th><th class="n">Proj.</th><th class="n">Fair</th><th class="n">DK line</th>
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
            src = f'<span class="cell-sub">DK {escape(r.as_of)}</span>' if known else ""
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
    n_bets = sum(1 for it in games for o in it["opts"] if o["ev"] >= C.MIN_EV)
    n_props = int(auto_props.flag.sum()) if auto_props is not None and not auto_props.empty else 0
    news_by_game = {}
    if news is not None and not news.empty:
        for r in news.fillna("").to_dict("records"):
            news_by_game[str(r.get("game", "")).strip()] = r
    td_by_game = {}
    if auto_props is not None and not auto_props.empty:
        td = auto_props[auto_props.stat == "anytime_td"].sort_values("adj_projection", ascending=False)
        td_by_game = {g: grp.head(5) for g, grp in td.groupby("game", sort=False)}
    props_by_game = {}
    if auto_props is not None and not auto_props.empty:
        props_by_game = {g: grp for g, grp in auto_props.groupby("game", sort=False)}
    key = lambda it: f"{it['g'].away_team} @ {it['g'].home_team}"
    cards = "".join(_game_card(it, news_by_game.get(key(it)), td_by_game.get(key(it)), props_by_game.get(key(it)), gi, injuries, market_info)
                    for gi, it in enumerate(games))
    tabs = [("games", "Games"), ("bets", "Best bets"), ("ratings", "Ratings"), ("backtest", "Track record"), ("how", "Glossary")]
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
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@700;800&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style></head>
<body>
<div class="mast"><div class="wrap">
  <div><p class="kicker">Week {week} · {season} NFL season</p><h1>The Rat Report <span class="beta">Beta</span></h1></div>
  <div class="mast-stats">
    <div><b>{n_bets}</b><span>game bets</span></div>
    <div><b>{n_props}</b><span>prop bets</span></div>
    <div><b>{updated}</b><span>updated (CT)</span></div>
  </div>
</div></div>
<nav class="toc" aria-label="Sections"><div class="wrap" role="tablist">{''.join(toc)}</div></nav>
<main class="wrap">
  <p class="beta-note"><b>Beta.</b> The model is still being tested: every flagged bet is tracked against the closing line under Track record, and features are kept only when they test better on past games. Treat picks as experimental.</p>
  <section id="bets"><h2>Best bets</h2>
    {checked_line}
    <div class="best">
      <div><h3>Games</h3>{_bets_section(games)}</div>
      <div><h3>Top 5 prop edges</h3>{_top_props(auto_props)}</div>
    </div>
    <p class="note">Prop lines are this week's DraftKings prices (newest available, never more than 3 days old), each with the time it was published; older lines are more likely to have moved, so check your app.{(' Last checked ' + escape(lines_checked) + '.') if lines_checked else ''}
    Prop projections adjust for the opponent's defense and expected pace, and blend with the DraftKings line; their replay on past weeks is under Track record.</p>
    <p class="note">Edge is the expected profit per dollar bet: +5% means about $5 back per $100 over many bets. Bets need at least a {C.MIN_EV:.0%} edge; smaller positive edges are leans.
    Hover or long-press a win % to see the break-even rate.</p></section>
  <section id="games"><h2>Games</h2>
    <p class="lede">Projected score beside each team. "Pure model" is the model alone; "Projection" blends it with the market.
    A gold bar marks a game with a bet. Each card has the most likely touchdown scorers, the top prop edges, every projected prop, and the latest news.</p>
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
      <div><b>Lean / No edge</b><p>A lean is a small positive edge below the betting bar. No edge means both sides are worse than the price.</p></div>
    </div></section>
</main>
<footer><div class="wrap">Generated {generated} from nflverse data through {dt}. Lines move all week, so rerun before betting.
For entertainment and analysis. Bet only what you can afford to lose. Help is available at 1-800-GAMBLER.</div></footer>
<script>{TABS_JS}</script>
</body></html>"""
