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

### Game-line model

```sh
betmap model backtest [--seasons 2015-2025] [--min-ev 0.02] [--model-weight 1.0]
betmap model predict                       # after `odds pull`
betmap odds scan --model-weight 0.25       # or the Model weight field on the Scan page
```

Team ratings from recency-weighted ridge regression on past margins and totals, turned into
probabilities with a normal distribution (whole-number lines can push). The backtest is
walk-forward: each week is predicted from earlier games only, then scored against nflverse
closing lines, with flat bets simulated at the closing price.

**Current result: no edge.** Over 2015-2025 the closing line predicts margins and totals
better than the model (margin MAE ~10.2 vs ~9.8), and betting the model's edges loses roughly
the vig, at every blend weight tried. The scan's model weight defaults to 0 (off) for that
reason. Treat a model edge as a reason to look closer, not to bet, until a backtest says
otherwise. The default half-life (150 days) was chosen on the same seasons it's evaluated on,
so even its small improvement is optimistic.

### Player prop model

```sh
betmap model backtest-props [--seasons 2024-2025]
betmap odds pull --markets "" --props player_reception_yds,player_rush_yds --days 2
betmap model predict                       # game lines and props in the latest pull
betmap odds scan --model-weight 0.25
betmap model evaluate                      # forward test, once games are final
```

For each player and stat: a recency-weighted average of his recent games, shrunk toward
his position's average, times how much the opponent allows to that position (also shrunk).
A negative binomial around that mean gives P(over); TDs and interceptions use a Poisson.
It models stats *given the player plays*, matching how books void props for inactive players.
Prop names are matched to nflverse by name and team; `predict` lists any it can't match.

**How it's tested.** nflverse has no historical prop lines, so the prop model can't be
backtested against real prices for free (The Odds API sells historical props from 2023).
Instead:

- `backtest-props` checks accuracy and calibration walk-forward, scoring only players with
  enough volume for a book to post the prop. Over 2024-2025 it beats a player's plain 8-game
  average on most markets (e.g. receiving yards MAE 23.2 vs 24.2) and its probabilities are
  roughly honest, though it underrates overs at the low end. Beating an average is a low bar;
  books' lines are much better than that.
- `evaluate` is the real test, for both models: every prediction stored before kickoff is
  compared with the devigged closing line once the game is final. It fills in as you pull
  odds (including props) during the season: run `model predict` after each pull, pull again
  within 6 hours of kickoff for the close, and `results sync` after the games. Until a
  market shows a lower Brier score than the close over a few hundred predictions, keep
  `--model-weight` low or at 0.

### Portfolio: joint risk and sizing

```sh
betmap portfolio risk                      # open bets simulated together, and overlaps
betmap portfolio size [--min-ev 0.02]      # size the scan's prices jointly
betmap portfolio correlations              # re-estimate the prop correlations
```

Also the **Portfolio** page in the web UI.

The scan sizes each bet as if it were the only one. Bets aren't independent, though: a team's
moneyline and spread win together, a QB's passing yards rise with his receivers', and an
Under hedges an Over. The portfolio models that with a Gaussian copula, where every bet wins
when a shared latent variable clears its probability's threshold:

- Game lines are exact functions of each game's margin and total. Team totals and first-half
  markets load on both, partially.
- Props link to their team's margin and the total ("game script"), to the same player's
  other stats, and to teammates. The links were estimated from nflverse 2021-2025: RB rushing
  yards rise when the team covers (+0.23), QB passing TDs with the total (+0.46),
  receptions with receiving yards (+0.77), a QB's yards with his receivers' (+0.29).
- Different games are independent.

`size` maximizes the expected log growth of everything together, with your open bets held
fixed, at your Kelly fraction and within `BETMAP_MAX_BET_FRACTION` per bet and
`BETMAP_MAX_GAME_FRACTION` per game (open bets count toward the game cap). Correlated bets
get smaller combined stakes than sizing them one at a time; a bet whose risk is already
covered gets 0. Open bets without a fair probability are assumed to be priced at a typical
margin.

## Roadmap

1. ✅ Ledger, bankroll, odds math (devig: multiplicative/power/Shin, EV, Kelly)
2. ✅ Odds API ingestion + best-price / devigged-consensus EV scan
3. ✅ nflverse stats sync, auto-settlement, CLV tracking
4. ✅ Game-line model + walk-forward backtest (no edge yet)
5. ✅ Player prop models + backtest (forward test accumulating)
6. ✅ Portfolio: joint simulation, overlap detection, correlated fractional Kelly
