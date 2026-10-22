# The Rat Report

Weekly NFL projections and bets from nflverse data. The live report is `index.html`, published with GitHub Pages and rebuilt every hour.

```
python run.py build-epa                      # one time: condense play-by-play into data/
python run.py fetch-props                    # live DraftKings player props -> prop_lines.csv
python run.py report --refresh               # this week's report -> output/report_<season>_wk<week>.html
```

`overrides.csv` corrects starting QBs (and DraftKings lines when confirmed) for upcoming games; `news.csv` is the per-game news check shown on the report.

`prop_lines.csv` comes from The Odds API (DraftKings book). Put your key in the `ODDS_API_KEY` environment variable or a git-ignored `.odds_api_key` file. Each refresh costs about 6 credits per game still to be played (about 80-90 for a full Sunday slate).

## Data the GitHub Actions keep fresh

| File | Action | Schedule |
|---|---|---|
| `prop_lines.csv`, `prop_books.csv` | Fetch DraftKings props (The Odds API, every US book) | Sun morning, Thu and Mon afternoons (free-plan sized) |
| `game_lines.csv` | Fetch game lines and weather | Sun/Mon/Wed/Fri mornings, 3 credits each |
| `weather.csv` | Fetch game lines and weather | Every 6 hours (free) |

```
python run.py fetch-props --hours 12   # props for games in the next 12 hours
python run.py fetch-games              # spreads/totals/moneylines at every book
python run.py fetch-weather            # kickoff forecasts, outdoor stadiums
python run.py build-extras             # red-zone usage + team tendencies from play-by-play
python run.py props-backtest           # replay past weeks of player projections
```

`bets_log.csv` is the closing-line tracker: every flagged bet, its price, the closing price, CLV and result.
