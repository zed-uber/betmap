from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from betmap.config import get_settings
from betmap.data import nflverse
from betmap.data.nflverse import StatsUnavailable
from betmap.db import init_db, make_engine, session_scope
from betmap.models.predict import load_predictions
from betmap.odds.client import OddsApiClient, OddsApiError
from betmap.odds.ingest import pull_odds
from betmap.odds.math import decimal_to_american, expected_value, kelly_fraction, parse_odds
from betmap.odds.scan import DEVIG_CHOICES, Opportunity, last_pull_at, scan
from betmap.portfolio.optimize import overlaps, risk, size
from betmap.portfolio.positions import candidate_positions, open_positions
from betmap.tables import Bet, BetStatus
from betmap.tracking import ledger
from betmap.tracking.results import Fetch, update_results

HERE = Path(__file__).parent
MARKET_TYPES = [
    "h2h",
    "spreads",
    "totals",
    "team_totals",
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]


def optional_float(value: str | None) -> float | None:
    return None if value is None or not value.strip() else float(value)


def redirect(path: str, anchor: str = "", **params: str) -> RedirectResponse:
    query = urlencode({k: v for k, v in params.items() if v})
    url = f"{path}?{query}" if query else path
    return RedirectResponse(f"{url}#{anchor}" if anchor else url, status_code=303)


def log_url(o: Opportunity, stake: float | None) -> str:
    """Dashboard link with the bet form prefilled from a scan opportunity."""
    params = {
        "event": o.event_label,
        "market": o.market_type,
        "selection": o.selection,
        "line": "" if o.line is None else f"{o.line:g}",
        "odds": f"{decimal_to_american(o.price):+.0f}",
        "book": o.book,
        "fair_prob": f"{o.fair_prob:.4f}",
        "stake": "" if stake is None else f"{stake:.2f}",
        "market_id": str(o.market_id),
    }
    return "/?" + urlencode(params)


class FieldError(ValueError):
    """Bad input in one form field; `field` is the input's name."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def parse_field[T](field: str, parse: Callable[[str], T], value: str, message: str) -> T:
    try:
        return parse(value)
    except ValueError:
        raise FieldError(field, message) from None


def parse_prob(value: str) -> float | None:
    """Accept 0.54, 54, or 54%; None when blank."""
    prob = optional_float(value.strip().removesuffix("%"))
    if prob is not None and prob > 1:
        prob /= 100
    if prob is not None and not 0 < prob < 1:
        raise ValueError(prob)
    return prob


def parse_stake(value: str) -> float:
    stake = float(value)
    if stake <= 0:
        raise ValueError(stake)
    return stake


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


def create_app(
    engine: Engine | None = None,
    odds_client: Callable[[], OddsApiClient] | None = None,
    results_fetch: Fetch | None = None,
) -> FastAPI:
    engine = engine or make_engine()
    results_fetch = results_fetch or nflverse.fetch
    odds_client = odds_client or (lambda: OddsApiClient(get_settings().odds_api_key))
    init_db(engine)

    app = FastAPI(title="betmap")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["american"] = lambda d: f"{decimal_to_american(d):+.0f}"
    templates.env.filters["money"] = lambda v: f"{v:,.2f}"
    templates.env.filters["signed"] = lambda v: f"{v:+,.2f}"
    templates.env.filters["pct"] = lambda v: f"{v:+.1%}"
    templates.env.filters["local"] = lambda dt, fmt="%a %-I:%M %p": dt.astimezone().strftime(fmt)
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
                "prefill": request.query_params,
                "error": error,
                "msg": msg,
            },
        )

    @app.get("/scan", response_class=HTMLResponse)
    def scan_page(
        request: Request,
        session: SessionDep,
        min_ev: float = 0.01,
        min_books: int = 3,
        method: str = "power",
        market: str = "",
        all_books: bool = False,
        model_weight: float = 0.0,
        error: str = "",
        msg: str = "",
    ):
        settings = get_settings()
        opportunities = scan(
            session,
            min_ev=min_ev,
            min_books=min_books,
            method=method if method in DEVIG_CHOICES else "power",
            market_type=market or None,
            books=None if all_books else settings.book_set,
            kelly_mult=settings.kelly_fraction,
            max_bet_fraction=settings.max_bet_fraction,
            model_probs=load_predictions(session) if model_weight else None,
            model_weight=min(max(model_weight, 0.0), 1.0),
        )
        equity = ledger.summarize(session).equity
        rows = []
        for o in opportunities:
            stake = round(o.kelly * equity, 2) if equity > 0 else None
            rows.append({"o": o, "stake": stake, "log_url": log_url(o, stake)})
        return templates.TemplateResponse(
            request,
            "scan.html",
            {
                "rows": rows,
                "last_pull": last_pull_at(session),
                "has_key": bool(settings.odds_api_key),
                "book_filter": sorted(settings.book_set),
                "markets": MARKET_TYPES,
                "methods": DEVIG_CHOICES,
                "kelly_mult": settings.kelly_fraction,
                "f": {
                    "min_ev": min_ev,
                    "min_books": min_books,
                    "method": method,
                    "market": market,
                    "all_books": all_books,
                    "model_weight": model_weight,
                },
                "error": error,
                "msg": msg,
            },
        )

    @app.get("/portfolio", response_class=HTMLResponse)
    def portfolio_page(
        request: Request,
        session: SessionDep,
        min_ev: float = 0.02,
        model_weight: float = 0.0,
        all_books: bool = False,
    ):
        settings = get_settings()
        existing, skipped = open_positions(session)
        equity = ledger.summarize(session).equity
        opportunities = scan(
            session,
            min_ev=min_ev,
            books=None if all_books else settings.book_set,
            kelly_mult=settings.kelly_fraction,
            max_bet_fraction=settings.max_bet_fraction,
            model_probs=load_predictions(session) if model_weight else None,
            model_weight=min(max(model_weight, 0.0), 1.0),
        )
        sized = size(
            candidate_positions(session, opportunities),
            existing,
            equity,
            kelly_mult=settings.kelly_fraction,
            max_bet=settings.max_bet_fraction,
            max_game=settings.max_game_fraction,
        )
        sized.sort(key=lambda z: (-z.portfolio, -z.independent))
        rows = []
        for z in sized:
            stake = round(z.portfolio * equity, 2) if equity > 0 else None
            rows.append({"z": z, "log_url": log_url(z.position.opportunity, stake)})
        return templates.TemplateResponse(
            request,
            "portfolio.html",
            {
                "existing": existing,
                "skipped": skipped,
                "risk": risk(existing) if existing else None,
                "overlaps": overlaps(existing + [z.position for z in sized if z.portfolio]),
                "rows": rows,
                "equity": equity,
                "settings": settings,
                "f": {"min_ev": min_ev, "model_weight": model_weight, "all_books": all_books},
            },
        )

    @app.post("/odds/pull")
    def pull(session: SessionDep):
        try:
            client = odds_client()
            n_events, n_snaps = pull_odds(session, client)
        except (OddsApiError, httpx.HTTPError) as e:
            return redirect("/scan", error=f"Couldn't pull odds: {e}")
        remaining = client.quota.remaining
        msg = f"Stored {n_snaps} prices across {n_events} games ({client.credits_spent} credits"
        msg += f", {remaining} left)" if remaining is not None else ")"
        return redirect("/scan", msg=msg)

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
        market_id: Annotated[str, Form()] = "",
    ):
        entered = {
            "event": event,
            "market": market,
            "selection": selection,
            "odds": odds,
            "stake": stake,
            "book": book,
            "line": line,
            "fair_prob": fair_prob,
            "notes": notes,
            "market_id": market_id,
        }
        try:
            price = parse_field(
                "odds",
                parse_odds,
                odds,
                "odds must be American (-110), decimal (1.91), or a contract price (0.57 / 57c)",
            )
            stake_amount = parse_field(
                "stake", parse_stake, stake, "stake must be a positive number"
            )
            line_value = parse_field(
                "line", optional_float, line, "line must be a number like -2.5"
            )
            prob = parse_field(
                "fair_prob", parse_prob, fair_prob, "fair prob must be like 0.54, 54, or 54%"
            )
            bet = ledger.place_bet(
                session,
                event_label=event.strip(),
                market_type=market,
                selection=selection.strip(),
                line=line_value,
                book=book.strip(),
                price=price,
                stake=stake_amount,
                fair_prob=prob,
                notes=notes.strip() or None,
                market_id=int(market_id) if market_id.strip() else None,
            )
        except ValueError as e:
            # Send back what was typed so the form can be corrected rather than retyped.
            invalid = e.field if isinstance(e, FieldError) else ""
            return redirect(
                "/", anchor="log", error=f"Couldn't log bet: {e}", invalid=invalid, **entered
            )
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

    @app.post("/results/sync")
    def sync_results(session: SessionDep):
        try:
            update = update_results(session, fetch=results_fetch)
        except (StatsUnavailable, httpx.HTTPError, OSError) as e:
            return redirect("/", error=f"Couldn't sync results: {e}")
        if update.needs_manual:
            return redirect("/", error=update.summary())
        return redirect("/", msg=update.summary())

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
