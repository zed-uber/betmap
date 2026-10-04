# betmap

NFL betting tracker, EV finder, and portfolio optimizer (game lines and player props).

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # or: uv sync --extra dev
cp .env.example .env   # add BETMAP_ODDS_API_KEY from the-odds-api.com
.venv/bin/betmap init
```

## Running

`betmap` is installed into the project venv, so activate it first:

```sh
source .venv/bin/activate
```

Web dashboard:

```sh
betmap web            # add --reload while developing
```

Then open http://127.0.0.1:8000. Stop it with Ctrl+C. It listens on localhost only and has no
login; pass `--host 0.0.0.0` to reach it from other devices on a trusted network.

The CLI and web UI share the same SQLite database (`data/betmap.db` by default), so bets logged
in one show up in the other.

## Usage

```sh
betmap bankroll deposit 1000
betmap bet add --event "KC @ BUF" --market spreads --selection BUF --line -2.5 \
    --odds -105 --stake 50 --book dk --fair-prob 0.54
betmap bet list [--all]
betmap bet settle 1 win --closing-odds -120
betmap report
```

Odds accept American (`-110`, `+150`), decimal (`1.91`), or a prediction-market contract price
(`0.57`, `57c`: cost to win $1, so 1/0.57 = 1.754 decimal). All are stored as decimal. Contract
prices are taken before fees; for Kalshi-style per-contract fees, add the fee to the price
(e.g. 57c + 2c fee → `59c`).

Fair prob accepts `0.54`, `54`, or `54%`. In the web form, a rejected entry keeps what you typed
and highlights the field to fix.

### Finding +EV prices

```sh
betmap odds pull                                  # moneyline/spread/total, next 7 days (3 credits)
betmap odds pull --markets "" --props player_pass_yds,player_rush_yds   # props: credits x events
betmap odds scan [--min-ev 0.02] [--market spreads] [--all-books]
```

Each book's price is scored against the devigged consensus of the *other* books quoting
both sides at the same line (at least `--min-books`, default 3). Stakes are fractional Kelly on
equity, capped at `BETMAP_MAX_BET_FRACTION`. Set `BETMAP_BOOKS=draftkings,fanduel,...` to only
show books you can bet at; every book still feeds the consensus.

The web **Scan** page does the same, with a button to pull game lines and a **Log** link on each
row that prefills the bet form (and links the bet to its market for CLV tracking later).

### Results, auto-settlement, and CLV

Needs the stats extra: `pip install -e '.[stats]'`.

```sh
betmap results sync [--dry-run] [--seasons 2025,2026]
```

This (or **Sync results** on the dashboard) pulls schedules, final scores, and player box
scores from nflverse, records closing lines, and settles open bets whose games are final:

- Bets are matched to games by their linked market or by the event label (`IND @ WAS`,
  `IND at WAS`), choosing the meeting nearest to when the bet was logged.
- Graded automatically: moneyline (a tie is a push), spreads, totals, `team_totals`
  (selection like `WAS Over`), alternate lines, and player props for passing, rushing, and
  receiving stats and anytime TD (selection like `Josh Allen Over` / `Josh Allen Yes`).
- Never guessed: a player missing from the box score (a DNP looks the same as a name typo),
  a game total far from the books' line (probably a team or half total), or a market without a
  grader. These are listed for you to settle by hand. Box scores usually land the morning
  after a game.

**CLV** is the EV your price had at the closing fair probability: the devigged consensus
of all books in the last odds pull before kickoff. It's only recorded if that pull was within
6 hours of kickoff, so pull odds shortly before games if you want CLV. For example, as cron
entries (Sunday 12:30 ET, plus prime-time games):

```cron
30 12 * * 0    cd ~/code/betmap && .venv/bin/betmap odds pull --days 0.5
0  19 * * 0,1,4 cd ~/code/betmap && .venv/bin/betmap odds pull --days 0.25
```

## Roadmap

1. ✅ Ledger, bankroll, odds math (devig: multiplicative/power/Shin, EV, Kelly)
2. ✅ Odds API ingestion + best-price / devigged-consensus EV scan
3. ✅ nflverse stats sync, auto-settlement, CLV tracking
4. Game-line model + walk-forward backtest
5. Player prop models + backtest
6. Portfolio: joint simulation, overlap detection, correlated fractional Kelly
