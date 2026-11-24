# Weekly review: 2026 week 11 (run Tue Nov 24, 10:17 AM CT)

## Props vs DraftKings lines
| stat       |   lines |   over_hit |   picked_over |   ll_ours |   ll_market |   leans |   lean_roi |
|:-----------|--------:|-----------:|--------------:|----------:|------------:|--------:|-----------:|
| pass_tds   |      22 |      0.5   |         0.273 |    0.6876 |      0.6762 |       3 |     -0.117 |
| pass_yds   |      23 |      0.565 |         0.522 |    0.7007 |      0.6935 |       4 |     -1     |
| rec_yds    |     116 |      0.448 |         0.328 |    0.6931 |      0.6931 |      58 |     -0.054 |
| receptions |     124 |      0.468 |         0.298 |    0.6756 |      0.6736 |       2 |     -0.103 |
| rush_yds   |      57 |      0.491 |         0.509 |    0.7079 |      0.6906 |      38 |     -0.302 |

## Anytime TDs
|   lines |   scored |   ll_model |   ll_market |   ll_blend |
|--------:|---------:|-----------:|------------:|-----------:|
|     168 |    0.232 |     0.4644 |      0.4564 |     0.4564 |

## Anytime TDs by DraftKings price (longshot check)
| band          |   lines |   scored |   dk_implied |   market_x092 |   model |   blend |
|:--------------|--------:|---------:|-------------:|--------------:|--------:|--------:|
| fav to +100   |      21 |       15 |        0.596 |         0.548 |   0.512 |   0.543 |
| +101 to +300  |      66 |       16 |        0.33  |         0.303 |   0.281 |   0.301 |
| +301 to +600  |      62 |        8 |        0.19  |         0.175 |   0.171 |   0.175 |
| +601 to +1000 |      11 |        0 |        0.114 |         0.105 |   0.107 |   0.105 |
| +1001 and up  |       8 |        0 |        0.073 |         0.067 |   0.124 |   0.07  |

## Bets and leans
| tier   | kind             |   n |   wins |   losses |   units |   clv |
|:-------|:-----------------|----:|-------:|---------:|--------:|------:|
| bet    | moneyline        |   3 |      1 |        2 |   -0.5  | 10.12 |
| bet    | prop: anytime_td |   4 |      1 |        3 |    1.8  |  4.19 |
| bet    | prop: pass_tds   |   2 |      0 |        2 |   -2    |  1.15 |
| bet    | prop: rec_yds    |  20 |     11 |        9 |    0.75 |  5.39 |
| bet    | prop: rush_yds   |  10 |      4 |        6 |   -2.48 |  5.53 |
| bet    | spread           |   3 |      0 |        3 |   -3    |  8.12 |
| lean   | moneyline        |  12 |      3 |        9 |   -4.9  |  6.03 |
| lean   | prop: anytime_td |  38 |      5 |       33 |   -6.9  |  1.71 |
| lean   | prop: pass_tds   |   6 |      2 |        4 |   -1.79 | -0.21 |
| lean   | prop: pass_yds   |  11 |      5 |        6 |   -1.57 |  3.32 |
| lean   | prop: rec_yds    | 113 |     62 |       51 |    4.02 |  3.3  |
| lean   | prop: receptions |  75 |     35 |       40 |   -4.6  |  4.08 |
| lean   | prop: rush_yds   |  57 |     27 |       30 |   -6.02 |  4.06 |
| lean   | spread           |   9 |      4 |        5 |   -1.26 |  1.78 |
| lean   | total            |  12 |      9 |        3 |    5.36 |  4.27 |

## Moneylines: favorites and +300 or longer (season; favorites are leans only since Nov 3)
| tier          |   n |   wins |   losses |   units |   clv |   avg_odds |
|:--------------|----:|-------:|---------:|--------:|------:|-----------:|
| +300 bet      |   7 |      0 |        6 |   -6    |  7.51 |        410 |
| +300 lean     |  15 |      0 |       12 |  -12    |  2.65 |        390 |
| favorite bet  |   9 |      4 |        4 |   -1.09 |  2.05 |       -142 |
| favorite lean |  35 |     20 |       11 |    1.03 |  1.23 |       -148 |

## Favorite-ML trial (bets 2+ days before kickoff, since Nov 17; ends if losing after 25 or beating the close <50%)
|   bets |   graded |   wins |   units |   beat_close |
|-------:|---------:|-------:|--------:|-------------:|
|      2 |        1 |      0 |      -1 |            1 |

## First TD model (leans only; season, at the first price shown)
| group                               |   n |   wins |   units |   sized |   hit |   exp |   beat_close |
|:------------------------------------|----:|-------:|--------:|--------:|------:|------:|-------------:|
| Values (price beats our fair odds)  |  32 |      0 |   -32   |    -3.7 | 0     | 0.063 |        0.344 |
| Top 4 per game by gap over the book |  52 |      5 |   -10.8 |     0.9 | 0.096 | 0.085 |        0.135 |
| Model's top pick per team           |  26 |      5 |     8.2 |     2.2 | 0.192 | 0.126 |        0.385 |
| Every player logged                 | 169 |     10 |   -68.2 |    -5.4 | 0.059 | 0.06  |        0.201 |
Hit rate vs Expected shows calibration; Beat close = first price longer than the last pre-game price.

## Market-move alarm: props by DraftKings' move against our side before logging (season)
| move            |   n |   weeks |   units |    roi |   clv |
|:----------------|----:|--------:|--------:|-------:|------:|
| 3-6 pts against | 104 |       4 |    -5.3 | -0.051 |  1.41 |
| 6+ pts against  | 109 |       4 |   -14.3 | -0.131 |  0.28 |
| other           | 689 |       4 |   -41.7 | -0.061 |  2.03 |

## Receiving/rushing yards bets since Week 9 (40% model weight)
| band   |   bets |   graded |   wins |   units |   clv |
|:-------|-------:|---------:|-------:|--------:|------:|
| 3-5%   |    101 |       78 |     44 |    4.91 |  5.93 |
| 5-8%   |     38 |       31 |     21 |    8.56 |  8.29 |
| 8%+    |     14 |       13 |      6 |   -1.73 |  8.99 |

## Bet size: flat 1u vs suggested size
| span      | tier       | group     |    n |   flat_u |   sized_risked |   sized_u |    roi |
|:----------|:-----------|:----------|-----:|---------:|---------------:|----------:|-------:|
| this week | bet        | all       |   42 |    -5.44 |           38.5 |     -6    | -0.129 |
| this week | lean       | all       |  333 |   -17.65 |          301.2 |     -9    | -0.053 |
| season    | bet        | all       |  438 |   -41.13 |          361.5 |    -21.74 | -0.094 |
| season    | lean       | all       | 1484 |   -25.16 |         1325.3 |      6.55 | -0.017 |
| season    | bets+leans | edge 0-3% | 1235 |   -22.49 |         1109   |     -2.67 | -0.018 |
| season    | bets+leans | edge 3-5% |  372 |   -33.07 |          333.2 |     -9.89 | -0.089 |
| season    | bets+leans | edge 5-8% |  180 |     2    |          148.3 |     -0.55 |  0.011 |
| season    | bets+leans | edge 8%+  |  132 |   -11.72 |           93.3 |     -1.06 | -0.089 |

## DraftKings vs Pinnacle (sharp-book check)
|   games |   ll_dk |   ll_pin |   spread_disagree |   pin_side_wins |   pushes |
|--------:|--------:|---------:|------------------:|----------------:|---------:|
|      13 |  0.5304 |   0.5345 |                 7 |               4 |        0 |

## DraftKings vs Pinnacle on props (same line)
|   props |   ll_dk |   ll_pin |
|--------:|--------:|---------:|
|     339 |  0.6413 |   0.6418 |

## Flags to investigate
- pass_tds: picks went 27% Over on 22 lines (Overs hit 50%)
- receptions: picks went 30% Over on 124 lines (Overs hit 47%)
- rush_yds: our chance was less accurate than DraftKings' (log loss 0.7079 vs 0.6906)

## Analyst notes (Tue Nov 24, 2026, weekly review run)
- Monday night: Commanders 22, Bengals 20 (nflverse). No bets in the game; leans 14-7 (WAS +6 won, CIN ML -275 lost, Under 52.5 won).
- Week 11 bets 17-25 (1 void), -5.4u, average CLV +5.3%: spreads 0-3, MLs 1-2, rec yds 11-9, rush yds 4-6 (all Unders), pass TDs 0-2, TDs 1-3 (+1.8u, Justice Hill +480).
- Real-line harness (replayed projections, last pre-kickoff DraftKings line; 2025 wk1-18 = 6,437 lines, 2026 wk1-7 = 2,627, live wk8-11 = 1,492):
  current log loss ours/DK: 2025 rush 0.6923/0.6929, 2026 wk1-7 0.6900/0.6935, live wk8-11 0.6969/0.6924; pass TDs 0.6819/0.6816, 0.6501/0.6556, 0.6810/0.6801; pass yds 0.6934/0.6929, 0.6983/0.6933, 0.6986/0.6932.
- Rushing yards in 2026 go Under DraftKings' lines (Over hit 44% wk1-7, 43% wk8-11 vs 49% in 2025).
- Fix tests (all OFF):
  1. Rush yds model weight 0.40 -> 0.25: 2025 0.6923 -> 0.6920, 2026 wk1-7 0.6900 -> 0.6908, live 0.6969 -> 0.6947 (mixed; 0.25 makes no rush bets). 0.10: 0.6923 / 0.6922 / 0.6931.
  2. Same-line FanDuel price bets need edge at DraftKings' price (>= 0%): closing-line bets 2025 +3.6u (240) -> -1.6u (221); 2026 wk1-7 +14.1u (108) -> +12.3u (103); live +3.7u (49) -> +4.1u (45). Worse in 2 of 3.
  3. Season-to-date Over-rate correction (shift by prior weeks' Over hit minus DK no-vig, shrunk N0=100-800): rush better in 2026 both periods (live 0.6969 -> 0.6884 at N0=100) but worse in 2025 at every N0 (0.6923 -> 0.6927-0.6939); rec yds/receptions mixed; pass yds/TDs worse.
- Pass TDs weight 0.1 vs 0.25 rechecked: 2025 0.6816 vs 0.6819, 2026 wk1-7 0.6532 vs 0.6501, live 0.6803 vs 0.6810 (mixed, unchanged).
- Moneyline calibration (walk-forward, closing no-vig, underdogs): 2025 n=256 model 31.6% DK 31.8% Pin 32.0% won 33.6%; 2026 wk1-7 n=98 33.8/34.2/34.3 vs 35.7; live wk8-11 n=54 30.6/31.0/30.7 vs 22.2. +300+: model 18.6% vs DK 18.1% vs won 19.4% (98). Log loss all dogs live: model 0.4982, DK 0.5024, Pin 0.5019. Model not too high on dogs.
- FanDuel line-shop props (bets, all weeks): line-shop 76-68 -0.7u CLV +2.3% (since 1.5% rule wk9-11: 44-30 +8.7u, CLV +4.2%); model-edge 4-3 +0.5u; same-line price 40-52 -8.6u CLV +0.5%. FD's off-market line vs DK's close: toward 295, same 161, away 54.
- TD bands on closing lines (2025 + 2026, DK): +601-1000 scored 8.6% vs DK 11.1% / x0.92 10.2% (1,412); +1001 up 5.4% vs 5.2% / 4.8% (2,607). Pinnacle two-sided at +601 and up: 139 lines (too few).
- Ops: GitHub's scheduled "Fetch DraftKings props" runs are firing every 4-7 hours instead of hourly (Nov 23-24: 13:41, 20:20, 01:14, 05:21, 11:38 UTC), so game lines go stale against the 3-hour health limit.
