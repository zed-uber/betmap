from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from betmap.config import get_settings
from betmap.tables import Base


def make_engine(db_path: Path | str | None = None) -> Engine:
    path = db_path if db_path is not None else get_settings().db_path
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    engine = engine or make_engine()
    init_db(engine)
    session = sessionmaker(engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
