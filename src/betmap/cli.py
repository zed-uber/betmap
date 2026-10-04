from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from betmap.config import get_settings
from betmap.db import init_db, make_engine, session_scope
from betmap.odds.math import decimal_to_american, expected_value, kelly_fraction, parse_odds
from betmap.tables import Bet, BetStatus
from betmap.tracking import ledger

app = typer.Typer(
    help="NFL betting tracker, EV finder, and portfolio optimizer.", no_args_is_help=True
)
bet_app = typer.Typer(help="Log, list, and settle bets.", no_args_is_help=True)
bankroll_app = typer.Typer(help="Deposits and withdrawals.", no_args_is_help=True)
app.add_typer(bet_app, name="bet")
app.add_typer(bankroll_app, name="bankroll")

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
    odds: Annotated[str, typer.Option(help="American (-110, +150) or decimal (1.91)")],
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
    if bet.closing_price:
        clv = bet.price / bet.closing_price - 1
        console.print(f"  CLV vs closing price: {clv:+.1%}")


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
    if summary.expected_profit_open is not None:
        table.add_row("Expected P/L on open", f"{summary.expected_profit_open:+.2f}")
    table.add_row("Available bankroll", f"{summary.bankroll:.2f}")
    table.add_row("Equity", f"{summary.equity:.2f}")
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
