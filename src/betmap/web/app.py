from collections.abc import Iterator
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from betmap.config import get_settings
from betmap.db import init_db, make_engine, session_scope
from betmap.odds.math import decimal_to_american, expected_value, kelly_fraction, parse_odds
from betmap.tables import Bet, BetStatus
from betmap.tracking import ledger

HERE = Path(__file__).parent
MARKET_TYPES = [
    "h2h",
    "spreads",
    "totals",
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]


def optional_float(value: str | None) -> float | None:
    return None if value is None or not value.strip() else float(value)


def redirect(path: str, **params: str) -> RedirectResponse:
    query = urlencode({k: v for k, v in params.items() if v})
    return RedirectResponse(f"{path}?{query}" if query else path, status_code=303)


def equity_curve(bets: list[Bet], width: int = 600, height: int = 120) -> dict | None:
    """Cumulative settled P/L as SVG polyline points."""
    settled = sorted((b for b in bets if b.settled_at), key=lambda b: b.settled_at)
    if len(settled) < 2:
        return None
    values = [0.0]
    for b in settled:
        values.append(values[-1] + b.profit)
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = width / (len(values) - 1)

    def y(v: float) -> float:
        return height - (v - lo) / span * height

    points = " ".join(f"{i * step:.1f},{y(v):.1f}" for i, v in enumerate(values))
    return {
        "points": points,
        "zero_y": y(0.0),
        "width": width,
        "height": height,
        "last": values[-1],
        "lo": lo,
        "hi": hi,
    }


def create_app(engine: Engine | None = None) -> FastAPI:
    engine = engine or make_engine()
    init_db(engine)

    app = FastAPI(title="betmap")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["american"] = lambda d: f"{decimal_to_american(d):+.0f}"
    templates.env.filters["money"] = lambda v: f"{v:,.2f}"
    templates.env.filters["signed"] = lambda v: f"{v:+,.2f}"
    templates.env.filters["pct"] = lambda v: f"{v:+.1%}"
    templates.env.globals["ev"] = lambda b: (
        None if b.fair_prob is None else expected_value(b.fair_prob, b.price)
    )

    def get_session() -> Iterator[Session]:
        with session_scope(engine) as s:
            yield s

    SessionDep = Annotated[Session, Depends(get_session)]

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, session: SessionDep, error: str = "", msg: str = ""):
        bets = session.scalars(select(Bet).order_by(Bet.placed_at.desc())).all()
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "summary": ledger.summarize(session),
                "open_bets": [b for b in bets if b.status == BetStatus.OPEN],
                "recent": [b for b in bets if b.status != BetStatus.OPEN][:10],
                "curve": equity_curve(list(bets)),
                "market_types": MARKET_TYPES,
                "results": [s.value for s in BetStatus if s != BetStatus.OPEN],
                "kelly_mult": get_settings().kelly_fraction,
                "error": error,
                "msg": msg,
            },
        )

    @app.get("/bets", response_class=HTMLResponse)
    def bets_page(request: Request, session: SessionDep, status: str = ""):
        query = select(Bet).order_by(Bet.placed_at.desc())
        if status:
            query = query.where(Bet.status == status)
        return templates.TemplateResponse(
            request,
            "bets.html",
            {
                "bets": session.scalars(query).all(),
                "status": status,
                "statuses": [s.value for s in BetStatus],
            },
        )

    @app.post("/bets")
    def add_bet(
        session: SessionDep,
        event: Annotated[str, Form()],
        market: Annotated[str, Form()],
        selection: Annotated[str, Form()],
        odds: Annotated[str, Form()],
        stake: Annotated[str, Form()],
        book: Annotated[str, Form()],
        line: Annotated[str, Form()] = "",
        fair_prob: Annotated[str, Form()] = "",
        notes: Annotated[str, Form()] = "",
    ):
        try:
            price = parse_odds(odds)
            prob = optional_float(fair_prob)
            if prob is not None and prob > 1:
                prob /= 100  # accept "54" as 54%
            bet = ledger.place_bet(
                session,
                event_label=event.strip(),
                market_type=market,
                selection=selection.strip(),
                line=optional_float(line),
                book=book.strip(),
                price=price,
                stake=float(stake),
                fair_prob=prob,
                notes=notes.strip() or None,
            )
        except ValueError as e:
            return redirect("/", error=f"Couldn't log bet: {e}")
        msg = f"Logged #{bet.id}: {bet.selection} {decimal_to_american(price):+.0f}"
        if prob is not None:
            kelly = kelly_fraction(prob, price) * get_settings().kelly_fraction
            msg += f" — EV {expected_value(prob, price):+.1%}, Kelly suggests {kelly:.2%}"
        return redirect("/", msg=msg)

    @app.post("/bets/{bet_id}/settle")
    def settle(
        session: SessionDep,
        bet_id: int,
        result: Annotated[str, Form()],
        closing_odds: Annotated[str, Form()] = "",
    ):
        try:
            closing = parse_odds(closing_odds) if closing_odds.strip() else None
            bet = ledger.settle_bet(session, bet_id, BetStatus(result), closing)
        except ValueError as e:
            return redirect("/", error=f"Couldn't settle bet: {e}")
        return redirect("/", msg=f"Settled #{bet.id} as {bet.status}: {bet.profit:+.2f}")

    @app.post("/bankroll")
    def transfer(
        session: SessionDep,
        kind: Annotated[str, Form()],
        amount: Annotated[str, Form()],
        book: Annotated[str, Form()] = "",
    ):
        try:
            ledger.record_transfer(session, kind, float(amount), book.strip() or None)
        except ValueError as e:
            return redirect("/", error=f"Couldn't record {kind}: {e}")
        return redirect("/", msg=f"Recorded {kind} of {float(amount):,.2f}")

    return app
