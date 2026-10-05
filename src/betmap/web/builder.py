"""Builder and slates pages: pick legs from the board, build parlays, compare and place slates."""

from collections.abc import Callable, Iterator
from typing import Annotated
from urllib.parse import parse_qsl, urlencode

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from betmap.builder import slates as sl
from betmap.config import get_settings
from betmap.odds.math import parse_odds
from betmap.odds.scan import BoardEntry, board, last_pull_at
from betmap.tables import Slate, SlateItem, SlateLeg, SlateStatus
from betmap.tracking import ledger

BOARD_LIMIT = 250


def _board(session: Session) -> list[BoardEntry]:
    settings = get_settings()
    return board(session, books=settings.book_set, fees=settings.fee_rates)


def _evaluate(session: Session, slate: Slate, entries: list[BoardEntry]) -> sl.SlateView:
    settings = get_settings()
    return sl.evaluate_slate(
        session,
        slate,
        entries,
        ledger.summarize(session).equity,
        settings.kelly_fraction,
        settings.max_bet_fraction,
        settings.max_game_fraction,
    )


def _back(back: str, **params) -> RedirectResponse:
    """Return to the builder view the form came from, updating some query parameters."""
    path, _, query = back.partition("?")
    if path not in ("/builder", "/slates"):
        path, query = "/builder", ""
    merged = dict(parse_qsl(query)) | {k: str(v) for k, v in params.items() if v is not None}
    merged = {k: v for k, v in merged.items() if v != ""}
    return RedirectResponse(f"{path}?{urlencode(merged)}" if merged else path, status_code=303)


def _filter(entries: list[BoardEntry], game: str, market: str, q: str, ev: str) -> list[BoardEntry]:
    q = q.strip().lower()
    return [
        e
        for e in entries
        if (not game or e.event_label == game)
        and (not market or e.market_type == market)
        and (not q or q in e.selection.lower() or q in e.event_label.lower())
        and (not ev or e.ev > 0)
    ]


def add_builder_routes(
    app: FastAPI, templates: Jinja2Templates, get_session: Callable[[], Iterator[Session]]
) -> None:
    SessionDep = Annotated[Session, Depends(get_session)]

    def find(session: Session, model, id_: int):
        obj = session.get(model, id_)
        if obj is None:
            raise ValueError(f"no such {model.__name__.lower()} #{id_}")
        return obj

    @app.get("/builder", response_class=HTMLResponse)
    def builder_page(
        request: Request,
        session: SessionDep,
        slate: int | None = None,
        parlay: int | None = None,
        game: str = "",
        market: str = "",
        q: str = "",
        ev: str = "",
        error: str = "",
        msg: str = "",
    ):
        drafts = sl.draft_slates(session)
        current = session.get(Slate, slate) if slate else (drafts[0] if drafts else None)
        entries = _board(session)
        shown = _filter(entries, game, market, q, ev)
        view = _evaluate(session, current, entries) if current else None
        building = session.get(SlateItem, parlay) if parlay else None
        if building is not None and (current is None or building.slate_id != current.id):
            building = None
        return templates.TemplateResponse(
            request,
            "builder.html",
            {
                "slate": current,
                "drafts": drafts,
                "view": view,
                "building": building,
                "rows": shown[:BOARD_LIMIT],
                "hidden_rows": max(0, len(shown) - BOARD_LIMIT),
                "games": sorted({e.event_label for e in entries}),
                "markets": sorted({e.market_type for e in entries}),
                "f": {"game": game, "market": market, "q": q, "ev": ev},
                "back": str(request.url.path)
                + ("?" + request.url.query if request.url.query else ""),
                "last_pull": last_pull_at(session),
                "error": error,
                "msg": msg,
            },
        )

    @app.post("/slates")
    def new_slate(
        session: SessionDep,
        name: Annotated[str, Form()] = "",
        back: Annotated[str, Form()] = "/builder",
    ):
        try:
            slate = sl.create_slate(session, name)
        except ValueError as e:
            return _back(back, error=f"Couldn't create slate: {e}")
        return _back(back, slate=slate.id, parlay="", msg=f"Created slate '{slate.name}'")

    @app.post("/slates/{slate_id}/add")
    def add_leg(
        session: SessionDep,
        slate_id: int,
        market_id: Annotated[int, Form()],
        side: Annotated[str, Form()],
        line: Annotated[str, Form()] = "",
        kind: Annotated[str, Form()] = "straight",
        parlay: Annotated[str, Form()] = "",
        back: Annotated[str, Form()] = "/builder",
    ):
        key = (market_id, side, float(line) if line.strip() else None)
        try:
            slate = find(session, Slate, slate_id)
            entry = next((e for e in _board(session) if e.key == key), None)
            if entry is None:
                raise ValueError("that price is no longer on the board")
            if kind == "parlay":
                item = find(session, SlateItem, int(parlay)) if parlay else None
                item = sl.add_to_parlay(session, slate, entry, item)
                return _back(back, slate=slate.id, parlay=item.id,
                             msg=f"Added {entry.selection} to parlay ({len(item.legs)} legs)")  # fmt: skip
            sl.add_straight(session, slate, entry)
        except ValueError as e:
            return _back(back, error=f"Couldn't add: {e}")
        return _back(back, slate=slate.id, msg=f"Added {entry.selection} as a straight bet")

    @app.post("/slates/legs/{leg_id}/remove")
    def remove(session: SessionDep, leg_id: int, back: Annotated[str, Form()] = "/builder"):
        try:
            sl.remove_leg(session, find(session, SlateLeg, leg_id))
        except ValueError as e:
            return _back(back, error=f"Couldn't remove: {e}")
        return _back(back)

    @app.post("/slates/items/{item_id}")
    def update(
        session: SessionDep,
        item_id: int,
        stake: Annotated[str, Form()] = "",
        offered: Annotated[str, Form()] = "",
        book: Annotated[str, Form()] = "",
        back: Annotated[str, Form()] = "/builder",
    ):
        try:
            item = find(session, SlateItem, item_id)
            changes = {"stake": float(stake) if stake.strip() else None}
            if item.kind == "parlay":
                changes["offered_price"] = parse_odds(offered) if offered.strip() else None
                changes["book"] = book.strip().lower() or None
            sl.update_item(session, item, **changes)
        except ValueError as e:
            return _back(back, error=f"Couldn't update: {e}")
        return _back(back, msg="Saved")

    @app.post("/slates/{slate_id}/rename")
    def rename(
        session: SessionDep,
        slate_id: int,
        name: Annotated[str, Form()] = "",
        back: Annotated[str, Form()] = "/builder",
    ):
        slate = session.get(Slate, slate_id)
        if slate is None or not name.strip():
            return _back(back, error="A slate needs a name")
        slate.name = name.strip()
        return _back(back, msg=f"Renamed to '{slate.name}'")

    @app.post("/slates/{slate_id}/duplicate")
    def duplicate(session: SessionDep, slate_id: int, back: Annotated[str, Form()] = "/builder"):
        try:
            copy = sl.duplicate_slate(session, find(session, Slate, slate_id))
        except ValueError as e:
            return _back(back, error=str(e))
        return _back(back, slate=copy.id, parlay="", msg=f"Copied to '{copy.name}'")

    @app.post("/slates/{slate_id}/delete")
    def delete(session: SessionDep, slate_id: int, back: Annotated[str, Form()] = "/slates"):
        slate = session.get(Slate, slate_id)
        if slate is None:
            return _back(back, error=f"No slate #{slate_id}")
        session.delete(slate)
        return _back(back, slate="", parlay="", msg=f"Deleted '{slate.name}'")

    @app.post("/slates/{slate_id}/place")
    def place(session: SessionDep, slate_id: int, back: Annotated[str, Form()] = "/builder"):
        try:
            slate = find(session, Slate, slate_id)
            bets = sl.place_slate(session, _evaluate(session, slate, _board(session)))
        except ValueError as e:
            return _back(back, error=f"Couldn't place: {e}")
        total = sum(b.stake for b in bets)
        count = f"{len(bets)} bet" + ("" if len(bets) == 1 else "s")
        query = urlencode({"msg": f"Placed '{slate.name}': {count}, {total:,.2f} staked"})
        return RedirectResponse(f"/bets?{query}", status_code=303)

    @app.get("/slates", response_class=HTMLResponse)
    def slates_page(
        request: Request,
        session: SessionDep,
        error: str = "",
        msg: str = "",
    ):
        # ?compare=1&compare=2: evaluate those drafts side by side.
        compare = [
            int(v) for k, v in request.query_params.multi_items() if k == "compare" and v.isdigit()
        ]
        drafts = sl.draft_slates(session)
        placed = session.scalars(
            select(Slate)
            .where(Slate.status == SlateStatus.PLACED)
            .order_by(Slate.updated_at.desc())
            .limit(10)
        ).all()
        entries = _board(session) if compare else []
        views = [_evaluate(session, s, entries) for s in drafts if s.id in compare]
        return templates.TemplateResponse(
            request,
            "slates.html",
            {
                "drafts": drafts,
                "placed": placed,
                "views": views,
                "compare": compare,
                "back": str(request.url.path)
                + ("?" + request.url.query if request.url.query else ""),
                "error": error,
                "msg": msg,
            },
        )
