from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
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
    add_missing_columns(engine)


def add_missing_columns(engine: Engine) -> None:
    """Minimal migration: add nullable columns that the models have but the tables lack.

    create_all only creates missing tables, so new columns on existing tables need this.
    Anything beyond adding nullable columns (renames, drops, NOT NULL) needs a real migration.
    """
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if not column.nullable:
                    raise RuntimeError(f"can't add NOT NULL column {table.name}.{column.name}")
                ddl = column.type.compile(engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl}'))


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
