from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from betmap.config import get_settings
from betmap.data.nflverse import StatsUnavailable
from betmap.db import init_db, make_engine, session_scope
from betmap.odds.client import GAME_MARKETS, OddsApiClient, OddsApiError
from betmap.odds.ingest import pull_odds
from betmap.odds.math import decimal_to_american, expected_value, kelly_fraction, parse_odds
from betmap.odds.scan import last_pull_at, scan
from betmap.tables import Bet, BetStatus
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
    regions: Annotated[str, typer.Option(help="Odds API regions: us, us2, eu, ...")] = "us",
    days: Annotated[float, typer.Option(help="Only games kicking off within this many days")] = 7,
) -> None:
    """Fetch current odds from The Odds API and store a snapshot."""
    try:
        client = OddsApiClient(get_settings().odds_api_key)
        with session_scope() as s:
            n_events, n_snaps = pull_odds(
                s,
                client,
                markets=split_csv(markets),
                props=split_csv(props),
                regions=regions,
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
) -> None:
    """List +EV prices from the latest odds pull."""
    settings = get_settings()
    with session_scope() as s:
        pulled = last_pull_at(s)
        opps = scan(
            s,
            min_ev=min_ev,
            min_books=min_books,
            method=method,
            market_type=market,
            books=None if all_books else settings.book_set,
            kelly_mult=settings.kelly_fraction,
            max_bet_fraction=settings.max_bet_fraction,
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
