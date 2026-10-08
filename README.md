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
prices are taken before fees; for Kalshi, add its fee per contract to the price
(0.07 x 0.57 x 0.43 ≈ 1.7c, so 57c → `58.7c`).

Parlays are one bet with legs:

```sh
betmap bet parlay --leg "KC @ BUF|spreads|BUF|-2.5" --leg "NE @ NYJ|totals|Over|41.5" \
    --odds +264 --stake 10 --book fanduel
```

Each `--leg` is `event|market|selection|line` (an optional fifth part is the leg's own odds).
`--odds` is the parlay price the book quotes: the product of the legs for a cross-game
parlay, or the book's own price for a same-game parlay. Results sync grades every leg: a losing
leg loses the parlay right away; once all legs are in, pushed or void legs drop out of a
cross-game parlay (their odds are divided out of the payout). A push in a same-game parlay goes
to manual settling, because books reprice those. A parlay's CLV uses each leg's closing line,
combined with the portfolio correlation model when legs share a game. Legs can't be edited;
void the parlay and log it again.

Price a parlay before betting it:

```sh
betmap parlay price --leg "TB @ DAL|h2h|DAL" --leg "DET @ ARI|h2h|DET"
betmap parlay price --leg "TB @ DAL|spreads|DAL|-9.5" --leg "TB @ DAL|totals|Over|47.5" --odds +250
```

Legs are matched to upcoming games in the latest pull. The fair probability is the chance every
leg wins: each leg's consensus fair probability, combined with the portfolio correlation model
(legs in different games are independent, so it's their product). Cross-game parlays are priced
at the sportsbook paying the most for all the legs (exchanges don't sell parlays); same-game
parlays need `--odds` with the book's quoted price, since books price those themselves. The
"as if independent" figure shows how much correlation moved the fair price.

Fair prob accepts `0.54`, `54`, or `54%`. In the web form, a rejected entry keeps what you typed
and highlights the field to fix.

### Finding +EV prices

```sh
betmap odds pull                                  # moneyline/spread/total, next 7 days (3 credits)
betmap odds pull --markets "" --props player_pass_yds,player_rush_yds   # props: credits x events
betmap odds pull --regions us,us2                 # whole regions instead of the book list
betmap odds scan [--min-ev 0.02] [--market spreads] [--all-books]
```

**Which books.** Pulls request the books in `BETMAP_PULL_BOOKS` rather than a region. Every 10
books cost the same credits as one region, so the default list (FanDuel, DraftKings, BetMGM,
BetRivers, LowVig, Pinnacle, Kalshi, Polymarket, Novig, Underdog) costs what the `us` region
alone did while adding exchanges and pick'em. Kalshi stands in for Robinhood, whose sports
contracts are listed on exchanges like Kalshi; log those bets with book `kalshi`.

**Exchange fees.** Kalshi charges a taker fee of 0.07 x price x (1 - price) per $1 contract
(1.75c at 50c, less toward the extremes). The scan scores and reports exchange prices *net*
of that fee (marked "after fee"), so the odds it pre-fills when you log a bet already include
it. Set rates per exchange with `BETMAP_EXCHANGE_FEES` (some Kalshi sports series charge half).

**Pick'em (Underdog, PrizePicks).** These pay multipliers on multi-pick entries, so a single
pick's listed price isn't a bet you can place. Their prices are stored but left out of the
consensus and the scan until pick'em entries are modeled (see the planned features below).

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

### Bet builder and slates

The web **Builder** page is where you put a bet card together:

- **The board** lists every priced side and line for upcoming games (filter by game, market,
  team/player, or +EV only), with the best price, fair probability, EV, and how many books the
  fair price rests on.
- **Slates** are saved, named drafts (`Sunday main`, `SGP ideas`). Add a line as a straight bet
  (**+ Bet**) or start a parlay (**+ Parlay**) and keep adding legs to it (**+ Leg**).
  Cross-game parlays are priced automatically; for a same-game parlay, type the book's quoted
  price. Prices refresh from the latest pull; legs that moved or are no longer offered are
  flagged.
- Each slate shows suggested stakes from joint Kelly sizing (with your open bets counted; type
  a stake to override), plus expected P/L, spread, chance of a net loss, a bad-week outcome,
  expected bankroll growth, and bets that are linked to each other or to what you already hold.
- **Duplicate** a slate to try a variation, then compare them side by side on the Slates page;
  growth is the best single number to choose by. **Place** logs every item with a stake to your
  bets (parlays with their legs), ready for results sync. Each bet links back to the slate
  and item it came from (shown next to it on the Bets page) and records its **source**:
  `manual`, or the model that built the slate. The source is fixed once placed, so each
  model's suggestions can be scored on their own. Placed slates can't be deleted, since
  they're the record of their bets; only drafts can.

```sh
betmap slate list | show ID | compare ID ID ... | place ID
```

**Second opinions.** Each board row and slate leg also shows other sources' probability for
that side, green when it makes the best price +EV, with a count of how many agree (`3/4`).
A dashed outline marks a source more than 5 points from the consensus. They don't change fair
prices, EV, or stakes:

- **Pinnacle**: the sharpest sportsbook, devigged at that exact line (pulled in the default
  book list for no extra credits; it's an opinion, not a place to bet).
- **Exchanges**: Kalshi, Polymarket, and Novig devigged and averaged, skipping thin markets.
- **nfelo** ([greerreNFL/nfelo](https://github.com/greerreNFL/nfelo)): an open-source NFL
  model's win probability and projected spread (moneylines and spreads only). Downloaded at
  most daily into `data/cache` (`BETMAP_NFELO_URL`, empty to turn off). nfelo publishes each
  week's games during that week, and it leans toward the market by design. Its repo has no
  license, so use the data for your own analysis only.
- **betmap**: the game and prop models (`model predict`).

`betmap model predict` also records Pinnacle, exchange, and nfelo probabilities, so
`betmap model evaluate` scores each source against the closing line over time. Once one
consistently beats the close, it's a candidate to blend into fair prices.

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

## Planned features

- **Underdog pick'em entries and typed bets** ⚠️ *Needs detailed review before planning.*
  Score pick'em properly and widen what a slate can hold, in roughly this order:
  1. ✅ *Groundwork.* Placed bets link back to the slate and item they came from (today only a
     parlay's notes mention the slate, and straights keep nothing). Bets get a `source`
     column (`manual`, or the model that suggested them, e.g. `synergy-v1`), set when the
     bet is placed and never changed afterward, so each model's suggestions can be scored on
     their own.
  2. *Tags.* Key-value tags on slates and bets, for grouping and filtering:
     `scenario=shootout`, `window=sun-early`, `experiment=fade-public`. Stored in their own
     tables (`slate_tags`, `bet_tags`) with real foreign keys. A key can have several values,
     since one parlay can play two scenarios. Keys are lowercased and trimmed, and a short
     list of known keys feeds autocomplete without being enforced. Tags are copied onto bets
     when they're placed, so editing or deleting a slate never rewrites history. The rule:
     anything the code acts on (kind, source) gets a real column; tags are only for
     grouping. A bet's thesis is prose and goes in `notes`.
  3. *Typed bets.* A bet's kind (straight, parlay, same-game parlay, pick'em entry, and later
     a DFS lineup) decides how it's priced, graded, and simulated, with kind-specific detail
     stored alongside. A slate can then mix kinds: a set of Underdog entries, or a spread,
     a total, and a same-game parlay on one game.
  4. *Pick'em entries.* Model entry payouts (e.g. 2-pick 3x means each leg must hit ~57.7%
     to break even), including non-default multipliers, and power vs flex entries. Flex
     entries pay even when some picks miss, so `simulate_returns` must handle partial
     payouts, not just win-or-lose. Choose which legs to combine, using the portfolio's
     correlation model to prefer (or avoid) linked legs. Underdog prices are already pulled
     and stored for when this is built.
  5. *Scenario coverage.* In the builder, a grid with each scenario tag as a row and each
     bet as a column, with P&L in the cells, to show whether a package only pays in one
     narrow outcome. If scenarios later need outcome conditions (e.g. "total over 55 and
     both QBs over 300 yards"), promote them from tags to their own table.
  6. *Later: DFS lineups.* A DraftKings lineup is graded on fantasy points against a contest
     field, so its payout depends on other entries. It fits in the typed bet, but pricing it
     is a separate project.
- ~~Bet builder: full board, parlays (incl. same-game), saved slates, compare, place~~ ✅
- ~~Second opinions next to each bet (Pinnacle, exchanges, nfelo, betmap models)~~ ✅
