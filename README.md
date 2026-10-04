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

Odds accept American (`-110`, `+150`) or decimal (`1.91`) and are stored as decimal.

## Roadmap

1. ✅ Ledger, bankroll, odds math (devig: multiplicative/power/Shin, EV, Kelly)
2. Odds API ingestion + best-price / devigged-consensus EV scan
3. nflverse stats sync, auto-settlement, CLV tracking
4. Game-line model + walk-forward backtest
5. Player prop models + backtest
6. Portfolio: joint simulation, overlap detection, correlated fractional Kelly
