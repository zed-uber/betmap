from statistics import fmean
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from betmap.backtest.games import backtest
from betmap.backtest.props import backtest_props
from betmap.config import get_settings
from betmap.data import nflverse
from betmap.data.nflverse import StatsUnavailable
from betmap.db import init_db, make_engine, session_scope
from betmap.models.evaluate import evaluate
from betmap.models.game_model import ModelConfig
from betmap.models.predict import load_predictions, predict_props, predict_upcoming
from betmap.odds.client import GAME_MARKETS, OddsApiClient, OddsApiError
from betmap.odds.ingest import pull_odds
from betmap.odds.math import decimal_to_american, expected_value, kelly_fraction, parse_odds
from betmap.odds.scan import last_pull_at, pickem_quotes, scan
from betmap.portfolio.correlation import estimate_correlations
from betmap.portfolio.optimize import overlaps, risk, size
from betmap.portfolio.positions import candidate_positions, open_positions
from betmap.tables import Bet, BetStatus, Market
from betmap.tracking import ledger
from betmap.tracking.results import update_results

app = typer.Typer(
    help="NFL betting tracker, EV finder, and portfolio optimizer.", no_args_is_help=True
)
bet_app = typer.Typer(help="Log, list, and settle bets.", no_args_is_help=True)
bankroll_app = typer.Typer(help="Deposits and withdrawals.", no_args_is_help=True)
app.add_typer(bet_app, name="bet")
app.add_typer(bankroll_app, name="bankroll")
odds_app = typer.Typer(help="Pull odds and scan for +EV prices.", no_args_is_help=True)
app.add_typer(odds_app, name="odds")
results_app = typer.Typer(help="Game results, auto-settlement, and CLV.", no_args_is_help=True)
app.add_typer(results_app, name="results")
model_app = typer.Typer(help="Game-line model: backtest and predictions.", no_args_is_help=True)
app.add_typer(model_app, name="model")
portfolio_app = typer.Typer(
    help="Joint risk, overlapping bets, and correlated Kelly sizing.", no_args_is_help=True
)
app.add_typer(portfolio_app, name="portfolio")

console = Console()


def fmt_odds(decimal: float) -> str:
    american = decimal_to_american(decimal)
    return f"{american:+.0f} ({decimal:.3f})"


@app.command()
def init() -> None:
    """Create the database and check configuration."""
    settings = get_settings()
    init_db(make_engine())
    console.print(f"Database ready at [bold]{settings.db_path}[/]")
    if not settings.odds_api_key:
        console.print("[yellow]BETMAP_ODDS_API_KEY is not set; odds pulls will be unavailable.[/]")


@bankroll_app.command("deposit")
def deposit(amount: float, book: Annotated[str | None, typer.Option()] = None) -> None:
    with session_scope() as s:
        ledger.record_transfer(s, "deposit", amount, book)
    console.print(f"Deposited {amount:.2f}")


@bankroll_app.command("withdraw")
def withdraw(amount: float, book: Annotated[str | None, typer.Option()] = None) -> None:
    with session_scope() as s:
        ledger.record_transfer(s, "withdrawal", amount, book)
    console.print(f"Withdrew {amount:.2f}")


@bet_app.command("add")
def bet_add(
    event: Annotated[str, typer.Option(help='Event label, e.g. "KC @ BUF"')],
    market: Annotated[str, typer.Option(help="h2h, spreads, totals, player_pass_yds, ...")],
    selection: Annotated[str, typer.Option(help='e.g. "BUF", "Over", "Josh Allen Over"')],
    odds: Annotated[
        str, typer.Option(help="American (-110), decimal (1.91), or contract price (0.57, 57c)")
    ],
    stake: Annotated[float, typer.Option()],
    book: Annotated[str, typer.Option()],
    line: Annotated[float | None, typer.Option()] = None,
    fair_prob: Annotated[float | None, typer.Option(help="Your fair win probability, 0-1")] = None,
    notes: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Log a placed bet."""
    price = parse_odds(odds)
    with session_scope() as s:
        bet = ledger.place_bet(
            s,
            event_label=event,
            market_type=market,
            selection=selection,
            line=line,
            book=book,
            price=price,
            stake=stake,
            fair_prob=fair_prob,
            notes=notes,
        )
    console.print(f"Logged bet #{bet.id}: {selection} {fmt_odds(price)} for {stake:.2f} at {book}")
    if fair_prob is not None:
        ev = expected_value(fair_prob, price)
        kelly = kelly_fraction(fair_prob, price) * get_settings().kelly_fraction
        console.print(f"  EV {ev:+.1%}, fractional Kelly suggests {kelly:.2%} of bankroll")


@bet_app.command("list")
def bet_list(
    all_: Annotated[bool, typer.Option("--all", help="Include settled bets")] = False,
) -> None:
    with session_scope() as s:
        query = select(Bet).order_by(Bet.placed_at)
        if not all_:
            query = query.where(Bet.status == BetStatus.OPEN)
        bets = s.scalars(query).all()

    table = Table(title="Bets" if all_ else "Open bets")
    for col in (
        "ID",
        "Placed",
        "Event",
        "Market",
        "Selection",
        "Line",
        "Odds",
        "Stake",
        "Book",
        "EV",
        "Status",
        "P/L",
    ):
        table.add_column(col)
    for b in bets:
        ev = expected_value(b.fair_prob, b.price) if b.fair_prob is not None else None
        table.add_row(
            str(b.id),
            b.placed_at.strftime("%m-%d"),
            b.event_label,
            b.market_type,
            b.selection,
            "" if b.line is None else f"{b.line:g}",
            f"{decimal_to_american(b.price):+.0f}",
            f"{b.stake:.2f}",
            b.book,
            "" if ev is None else f"{ev:+.1%}",
            b.status,
            "" if b.profit is None else f"{b.profit:+.2f}",
        )
    console.print(table)


@bet_app.command("settle")
def bet_settle(
    bet_id: int,
    result: Annotated[BetStatus, typer.Argument(help="win, loss, push, or void")],
    closing_odds: Annotated[str | None, typer.Option(help="Closing price, for CLV")] = None,
) -> None:
    """Grade an open bet."""
    closing = parse_odds(closing_odds) if closing_odds else None
    with session_scope() as s:
        bet = ledger.settle_bet(s, bet_id, result, closing)
    console.print(f"Bet #{bet.id} settled as {bet.status}: {bet.profit:+.2f}")
    if bet.clv is not None:
        console.print(f"  CLV: {bet.clv:+.1%}")


@bet_app.command("edit")
def bet_edit(
    bet_id: int,
    event: Annotated[str | None, typer.Option()] = None,
    market: Annotated[str | None, typer.Option()] = None,
    selection: Annotated[str | None, typer.Option()] = None,
    line: Annotated[float | None, typer.Option()] = None,
    odds: Annotated[str | None, typer.Option(help="American, decimal, or contract price")] = None,
    stake: Annotated[float | None, typer.Option()] = None,
    book: Annotated[str | None, typer.Option()] = None,
    fair_prob: Annotated[float | None, typer.Option(help="0-1")] = None,
    notes: Annotated[str | None, typer.Option()] = None,
    status: Annotated[
        BetStatus | None, typer.Option(help="win, loss, push, void, or open to un-settle")
    ] = None,
) -> None:
    """Correct a bet. Only the options you pass change; a new status recomputes the payout."""
    changes = {
        "event_label": event,
        "market_type": market,
        "selection": selection,
        "line": line,
        "price": parse_odds(odds) if odds else None,
        "stake": stake,
        "book": book,
        "fair_prob": fair_prob,
        "notes": notes,
        "status": status,
    }
    changes = {k: v for k, v in changes.items() if v is not None}
    if not changes:
        console.print("Nothing to change; pass at least one option (see --help).")
        raise typer.Exit(1)
    try:
        with session_scope() as s:
            bet = ledger.edit_bet(s, bet_id, **changes)
            result = "open" if bet.status == BetStatus.OPEN else f"{bet.status}, {bet.profit:+.2f}"
            label = f"{bet.event_label} {bet.selection} {fmt_odds(bet.price)} for {bet.stake:.2f}"
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print(f"Bet #{bet_id}: {label} ({result})")


@app.command()
def report() -> None:
    """Bankroll and performance summary."""
    with session_scope() as s:
        summary = ledger.summarize(s)
    table = Table(show_header=False)
    table.add_row("Net deposits", f"{summary.deposits - summary.withdrawals:.2f}")
    table.add_row("Settled P/L", f"{summary.settled_profit:+.2f}")
    table.add_row("Record (W-L-P)", f"{summary.wins}-{summary.losses}-{summary.pushes}")
    table.add_row("ROI", "n/a" if summary.roi is None else f"{summary.roi:+.1%}")
    table.add_row("Open bets", f"{summary.open_bets} ({summary.open_exposure:.2f} at risk)")
    if summary.avg_clv is not None:
        table.add_row(
            "Avg CLV",
            f"{summary.avg_clv:+.1%} over {len(summary.clv_values)} bets, "
            f"{summary.beat_close:.0%} beat the close",
        )
    if summary.expected_profit_open is not None:
        table.add_row("Expected P/L on open", f"{summary.expected_profit_open:+.2f}")
    table.add_row("Available bankroll", f"{summary.bankroll:.2f}")
    table.add_row("Equity", f"{summary.equity:.2f}")
    console.print(table)


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


@odds_app.command("pull")
def odds_pull(
    markets: Annotated[str, typer.Option(help="Game markets, comma-separated")] = ",".join(
        GAME_MARKETS
    ),
    props: Annotated[
        str, typer.Option(help="Prop markets, e.g. player_pass_yds; costs credits per event")
    ] = "",
    bookmakers: Annotated[
        str, typer.Option(help="Books to pull (default BETMAP_PULL_BOOKS); 10 cost one region")
    ] = "",
    regions: Annotated[
        str, typer.Option(help="Pull whole regions instead: us, us2, us_ex, us_dfs, ...")
    ] = "",
    days: Annotated[float, typer.Option(help="Only games kicking off within this many days")] = 7,
) -> None:
    """Fetch current odds from The Odds API and store a snapshot."""
    settings = get_settings()
    books = () if regions else (split_csv(bookmakers) or settings.pull_book_list)
    try:
        client = OddsApiClient(settings.odds_api_key)
        with session_scope() as s:
            n_events, n_snaps = pull_odds(
                s,
                client,
                markets=split_csv(markets),
                props=split_csv(props),
                regions=regions or "us",
                bookmakers=books,
                days=days,
            )
    except OddsApiError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print(f"Stored {n_snaps} prices across {n_events} games")
    q = client.quota
    console.print(f"  Credits: {client.credits_spent} spent, {q.remaining} remaining this month")


@odds_app.command("scan")
def odds_scan(
    min_ev: Annotated[float, typer.Option(help="Minimum EV per unit staked")] = 0.01,
    min_books: Annotated[int, typer.Option(help="Books needed to form a consensus")] = 3,
    method: Annotated[str, typer.Option(help="Devig: power, multiplicative, shin")] = "power",
    market: Annotated[str | None, typer.Option(help="Only this market type")] = None,
    all_books: Annotated[bool, typer.Option("--all-books", help="Ignore BETMAP_BOOKS")] = False,
    model_weight: Annotated[
        float, typer.Option(help="Blend in the game model (0-1); run `model predict` first")
    ] = 0.0,
) -> None:
    """List +EV prices from the latest odds pull."""
    settings = get_settings()
    with session_scope() as s:
        pulled = last_pull_at(s)
        pickem = pickem_quotes(s)
        opps = scan(
            s,
            min_ev=min_ev,
            min_books=min_books,
            method=method,
            market_type=market,
            books=None if all_books else settings.book_set,
            kelly_mult=settings.kelly_fraction,
            max_bet_fraction=settings.max_bet_fraction,
            model_probs=load_predictions(s) if model_weight else None,
            model_weight=model_weight,
            fees=settings.fee_rates,
        )
        equity = ledger.summarize(s).equity
    if pulled is None:
        console.print("No odds yet; run [bold]betmap odds pull[/] first.")
        return
    table = Table(title=f"+EV prices (odds as of {pulled.astimezone():%a %H:%M})")
    for col in (
        "Kickoff",
        "Event",
        "Market",
        "Selection",
        "Line",
        "Book",
        "Odds",
        "Fair",
        "EV",
        "Books",
        "Stake",
    ):
        table.add_column(col)
    for o in opps:
        table.add_row(
            f"{o.kickoff.astimezone():%a %H:%M}",
            o.event_label,
            o.market_type,
            o.selection,
            ""
            if o.line is None
            else f"{o.line:+g}"
            if o.market_type.endswith("spreads")
            else f"{o.line:g}",
            o.book,
            f"{decimal_to_american(o.price):+.0f}",
            f"{decimal_to_american(1 / o.fair_prob):+.0f}",
            f"{o.ev:+.1%}",
            str(o.n_books),
            f"{o.kelly * equity:.2f}" if equity > 0 else f"{o.kelly:.2%}",
        )
    console.print(table if opps else "No prices above the EV threshold.")
    if any(o.fee_adjusted for o in opps):
        console.print("[dim]Exchange prices (e.g. Kalshi) are shown net of taker fees.[/]")
    if pickem:
        console.print(
            f"[dim]{pickem} pick'em prices (Underdog etc.) stored but not scored: pick'em pays "
            "on whole entries, not single picks.[/]"
        )


@results_app.command("sync")
def results_sync(
    seasons: Annotated[
        str, typer.Option(help="Comma-separated seasons; default is the current one")
    ] = "",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show grades without settling")
    ] = False,
) -> None:
    """Sync scores and box scores from nflverse, record closing lines, settle finished bets."""
    season_list = [int(s) for s in split_csv(seasons)] or None
    try:
        with session_scope() as s:
            update = update_results(s, seasons=season_list, dry_run=dry_run)
            rows = [
                (o.bet.id, o.bet.event_label, o.bet.selection, o.status, o.reason, o.needs_manual)
                for o in update.outcomes
            ]
    except StatsUnavailable as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print(
        f"Synced {update.games} games, {update.player_lines} player lines; "
        f"recorded {update.closing_lines} closing lines"
    )
    if not rows:
        console.print("No open bets.")
        return
    table = Table(title="Open bets" + (" (dry run)" if dry_run else ""))
    for col in ("ID", "Event", "Selection", "Result", "Note"):
        table.add_column(col)
    for bet_id, event, selection, status, reason, manual in rows:
        result = status or ("[yellow]manual[/]" if manual else "[dim]pending[/]")
        table.add_row(str(bet_id), event, selection, result, reason)
    console.print(table)


def parse_seasons(text: str) -> list[int]:
    """'2015-2025' or '2019,2021,2023'."""
    seasons: list[int] = []
    for part in split_csv(text):
        start, _, end = part.partition("-")
        seasons.extend(range(int(start), int(end or start) + 1))
    return seasons


def model_config(half_life: float, ridge: float, margin_sd: float, total_sd: float):
    return ModelConfig(
        half_life_days=half_life, ridge=ridge, margin_sd=margin_sd, total_sd=total_sd
    )


HalfLife = Annotated[float, typer.Option(help="Days for a game's weight to halve")]
Ridge = Annotated[float, typer.Option(help="Shrinkage toward an average team")]
MarginSd = Annotated[float, typer.Option(help="Std dev of margin around prediction")]
TotalSd = Annotated[float, typer.Option(help="Std dev of total around prediction")]
_defaults = ModelConfig()


@model_app.command("backtest")
def model_backtest(
    seasons: Annotated[str, typer.Option(help="Test seasons, e.g. 2015-2025")] = "2015-2025",
    train_years: Annotated[int, typer.Option(help="Years of history to fit on")] = 4,
    min_ev: Annotated[float, typer.Option(help="Bet when EV at the close is at least this")] = 0.02,
    model_weight: Annotated[
        float, typer.Option(help="Model share of the probability; the rest is the market")
    ] = 1.0,
    half_life: HalfLife = _defaults.half_life_days,
    ridge: Ridge = _defaults.ridge,
    margin_sd: MarginSd = _defaults.margin_sd,
    total_sd: TotalSd = _defaults.total_sd,
) -> None:
    """Walk-forward backtest against historical closing lines (nflverse)."""
    test = parse_seasons(seasons)
    try:
        rows = nflverse.fetch_schedules(list(range(min(test) - train_years, max(test) + 1)))
    except StatsUnavailable as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    history = [
        (nflverse.model_game(r), r["season"], r["week"], nflverse.closing_lines(r)) for r in rows
    ]
    report = backtest(
        history,
        test,
        model_config(half_life, ridge, margin_sd, total_sd),
        min_ev=min_ev,
        train_days=train_years * 365,
        model_weight=model_weight,
    )
    if not report.games:
        console.print("No completed games with closing lines in those seasons.")
        return

    fit = Table(title=f"Prediction error, {report.games} games (lower is better)")
    for col in ("", "Model", "Closing line"):
        fit.add_column(col)
    fit.add_row("Margin MAE", f"{report.margin_mae_model:.2f}", f"{report.margin_mae_market:.2f}")
    fit.add_row("Total MAE", f"{report.total_mae_model:.2f}", f"{report.total_mae_market:.2f}")
    if report.brier_model is not None:
        fit.add_row("Home win Brier", f"{report.brier_model:.4f}", f"{report.brier_market:.4f}")
    console.print(fit)

    bets = Table(title=f"Flat 1u bets at the closing price, EV >= {min_ev:.0%}")
    markets = ("spreads", "totals", "h2h")
    for col in ("Season", *markets):
        bets.add_column(col)

    def cell(r) -> str:
        if not r.bets:
            return "-"
        return f"{r.wins}-{r.losses}-{r.pushes}  {r.roi:+.1%}"

    for season in sorted(report.by_season):
        bets.add_row(str(season), *(cell(report.by_season[season][m]) for m in markets))
    bets.add_row("[bold]All[/]", *(cell(report.markets[m]) for m in markets))
    claimed = [
        f"{sum(r.claimed_ev) / len(r.claimed_ev):+.1%}" if r.bets else "-"
        for r in (report.markets[m] for m in markets)
    ]
    bets.add_row("[dim]claimed EV[/]", *claimed)
    console.print(bets)
    console.print(
        "[dim]At -110, breaking even needs 52.4% wins. If ROI is far below the claimed EV, "
        "the model is overconfident.[/]"
    )


@model_app.command("predict")
def model_predict(
    train_years: Annotated[int, typer.Option(help="Years of history to fit on")] = 4,
    half_life: HalfLife = _defaults.half_life_days,
    ridge: Ridge = _defaults.ridge,
    margin_sd: MarginSd = _defaults.margin_sd,
    total_sd: TotalSd = _defaults.total_sd,
) -> None:
    """Store model probabilities for upcoming lines (games and player props) in the latest pull."""
    season = nflverse.current_season()
    try:
        rows = nflverse.fetch_schedules(list(range(season - train_years, season + 1)))
        with session_scope() as s:
            n = predict_upcoming(s, rows, model_config(half_life, ridge, margin_sd, total_sd))
            has_props = s.scalar(select(Market.id).where(Market.player.is_not(None)).limit(1))
        console.print(f"Stored {n} game-line predictions")
        if has_props:
            # Two seasons covers each player's recent games and each defense's last 16.
            stats = nflverse.fetch_player_stats([season - 1, season])
            with session_scope() as s:
                n_props, unmatched = predict_props(s, stats)
            console.print(f"Stored {n_props} prop predictions")
            if unmatched:
                console.print(
                    f"[yellow]Couldn't match {len(unmatched)} players to nflverse:[/] "
                    + ", ".join(unmatched[:10])
                    + (" ..." if len(unmatched) > 10 else "")
                )
    except StatsUnavailable as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print("Use them with: betmap odds scan --model-weight 0.25")


@model_app.command("backtest-props")
def model_backtest_props(
    seasons: Annotated[str, typer.Option(help="Test seasons, e.g. 2024-2025")] = "2024-2025",
    train_years: Annotated[int, typer.Option(help="Seasons of history before the test")] = 2,
) -> None:
    """Walk-forward accuracy and calibration of the prop model (no lines needed)."""
    test = parse_seasons(seasons)
    try:
        rows = nflverse.fetch_player_stats(list(range(min(test) - train_years, max(test) + 1)))
    except StatsUnavailable as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    reports = backtest_props(rows, test)
    if not reports:
        console.print("No player games to test in those seasons.")
        return
    table = Table(title="Prop model vs a player's plain 8-game average (lower MAE is better)")
    for col in ("Market", "N", "MAE model", "MAE avg", "In 50%", "In 80%", "Calibration"):
        table.add_column(col)
    for market, r in sorted(reports.items()):
        calibration = "  ".join(
            f"{fmean(b.predicted):.0%}→{b.actual:.0%}" for b in r.buckets.values() if b.n >= 30
        )
        table.add_row(
            market.removeprefix("player_"),
            str(r.n),
            f"{r.mae_model:.2f}",
            f"{r.mae_baseline:.2f}",
            f"{r.in_50 / r.n:.0%}",
            f"{r.in_80 / r.n:.0%}",
            calibration,
        )
    console.print(table)
    console.print(
        "[dim]Calibration: predicted P(over) → how often it went over, against a stand-in "
        "line at the player's recent average. Intervals over-cover on low counts because "
        "stats are whole numbers. Beating a plain average is not beating the books: see "
        "`betmap model evaluate`.[/]"
    )


@model_app.command("evaluate")
def model_evaluate() -> None:
    """Forward test: stored pre-game predictions vs the closing line, on final games."""
    with session_scope() as s:
        result = evaluate(s)
    if not result.groups:
        console.print(
            "Nothing to evaluate yet. It needs `model predict` before kickoff, an odds pull "
            "within 6h of kickoff, and `results sync` after the game."
        )
    else:
        table = Table(title="Brier score vs the devigged closing line (lower is better)")
        for col in ("Model", "Market", "N", "Model", "Closing line", "Verdict"):
            table.add_column(col)
        for (model, market), g in sorted(result.groups.items()):
            better = g.brier_model < g.brier_market
            verdict = "beats close" if better else "worse than close"
            if g.n < 200:
                verdict += " (too few to trust)"
            table.add_row(
                model, market, str(g.n), f"{g.brier_model:.4f}", f"{g.brier_market:.4f}", verdict
            )
        console.print(table)
    console.print(
        f"[dim]{result.pending} waiting on results, {result.no_close} without a closing "
        "consensus to compare against.[/]"
    )


@portfolio_app.command("risk")
def portfolio_risk() -> None:
    """Simulated P/L of all open bets together, and which ones overlap."""
    with session_scope() as s:
        positions, skipped = open_positions(s)
        summary = risk(positions)
        linked = overlaps(positions)
    if not positions:
        console.print("No open bets to analyze.")
    else:
        table = Table(title=f"Open bets: {len(positions)} positions", show_header=False)
        table.add_row("Expected P/L", f"{summary.expected:+.2f}")
        table.add_row("Std dev", f"{summary.sd:.2f}")
        table.add_row("Chance of a net loss", f"{summary.p_loss:.0%}")
        table.add_row("Bad week (5th pct)", f"{summary.p05:+.2f}")
        table.add_row("Everything loses", f"{summary.worst:+.2f}")
        for game, staked in sorted(summary.by_game.items(), key=lambda kv: -kv[1]):
            table.add_row(f"  at risk on {game}", f"{staked:.2f}")
        console.print(table)
        assumed = [p.label for p in positions if p.prob_source == "assumed"]
        if assumed:
            console.print(
                f"[dim]No fair probability for {', '.join(assumed)}: assumed the price "
                "minus a typical margin.[/]"
            )
    if linked:
        table = Table(title="Overlapping bets")
        for col in ("Bet", "Bet", "Correlation"):
            table.add_column(col)
        for o in linked:
            table.add_row(o.a.label, o.b.label, f"{o.correlation:+.2f}")
        console.print(table)
    for bet in skipped:
        console.print(
            f"[yellow]Couldn't place #{bet.id} ({bet.event_label} {bet.selection}) in a game.[/]"
        )


@portfolio_app.command("size")
def portfolio_size(
    min_ev: Annotated[float, typer.Option(help="Candidates from the scan above this EV")] = 0.02,
    min_books: Annotated[int, typer.Option(help="Books needed to form a consensus")] = 3,
    model_weight: Annotated[float, typer.Option(help="Blend in model predictions (0-1)")] = 0.0,
    all_books: Annotated[bool, typer.Option("--all-books", help="Ignore BETMAP_BOOKS")] = False,
) -> None:
    """Size the scan's +EV prices together, accounting for correlation and open bets."""
    settings = get_settings()
    with session_scope() as s:
        opps = scan(
            s,
            min_ev=min_ev,
            min_books=min_books,
            books=None if all_books else settings.book_set,
            kelly_mult=settings.kelly_fraction,
            max_bet_fraction=settings.max_bet_fraction,
            model_probs=load_predictions(s) if model_weight else None,
            model_weight=model_weight,
            fees=settings.fee_rates,
        )
        candidates = candidate_positions(s, opps)
        existing, _ = open_positions(s)
        equity = ledger.summarize(s).equity
    if not candidates:
        console.print("No candidates; pull odds or lower --min-ev.")
        return
    if equity <= 0:
        console.print("[yellow]No bankroll recorded; stakes are shown as % of bankroll.[/]")
    sized = size(
        candidates,
        existing,
        equity,
        kelly_mult=settings.kelly_fraction,
        max_bet=settings.max_bet_fraction,
        max_game=settings.max_game_fraction,
    )
    table = Table(title="Suggested stakes (fractional Kelly, sized together)")
    for col in ("Game", "Bet", "Odds", "EV", "Alone", "Together"):
        table.add_column(col)

    def money(frac: float) -> str:
        return f"{frac * equity:.2f}" if equity > 0 else f"{frac:.2%}"

    for z in sorted(sized, key=lambda z: (-z.portfolio, -z.independent)):
        p = z.position
        table.add_row(
            p.game_label,
            p.label,
            f"{decimal_to_american(p.price):+.0f}",
            f"{p.prob * p.price - 1:+.1%}",
            money(z.independent),
            money(z.portfolio) if z.portfolio else "[dim]0[/]",
        )
    console.print(table)
    console.print(
        f"[dim]Alone = each bet sized by itself (what the scan shows). Together = joint "
        f"Kelly with your {len(existing)} open bets held fixed, max "
        f"{settings.max_bet_fraction:.0%} per bet and {settings.max_game_fraction:.0%} "
        "per game.[/]"
    )


@portfolio_app.command("correlations")
def portfolio_correlations(
    seasons: Annotated[str, typer.Option(help="Seasons to estimate from")] = "2021-2025",
) -> None:
    """Re-estimate the prop correlation numbers from nflverse (compare with the built-ins)."""
    years = parse_seasons(seasons)
    try:
        players = nflverse.fetch_player_stats(years)
        games = nflverse.fetch_schedules(years)
    except StatsUnavailable as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    est = estimate_correlations(players, games)
    table = Table(title="Game script: prop Over vs team margin and total surprise")
    for col in ("Market", "Margin", "Total"):
        table.add_column(col)
    for market, (m, t) in est["game_script"].items():
        table.add_row(market, f"{m:+.2f}", f"{t:+.2f}")
    console.print(table)
    table = Table(title="Same player, after game script")
    for col in ("Markets", "Correlation"):
        table.add_column(col)
    for pair, r in sorted(est["same_player"].items(), key=lambda kv: -abs(kv[1])):
        if abs(r) >= 0.1:
            table.add_row(" ~ ".join(sorted(m.removeprefix("player_") for m in pair)), f"{r:+.2f}")
    console.print(table)
    console.print(
        f"Teammates: QB pass yds ~ receiver yds {est['qb_to_receiver']:+.2f}, "
        f"anytime TD ~ anytime TD {est['td_to_td']:+.2f}"
    )
    console.print("[dim]Built-in values live in betmap/portfolio/correlation.py.[/]")


@app.command()
def web(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes")] = False,
) -> None:
    """Run the web dashboard."""
    import uvicorn

    uvicorn.run("betmap.web.app:create_app", factory=True, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
