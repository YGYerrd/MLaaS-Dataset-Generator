from __future__ import annotations
import sqlite3, os
from typing import Mapping, Any

def make_writer(kind: str, **kwargs):
    if kind == "sqlite":
        return SQLiteWriter(**kwargs)
    raise ValueError(f"Unknown writer kind: {kind}")

class SQLiteWriter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def start(self) -> None:
        folder = os.path.dirname(self.db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
            
        init_needed = not os.path.exists(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        if init_needed:
            from importlib import resources
            sql = resources.files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")
            self.conn.executescript(sql)
            self.conn.commit()

    def _ins(self, table: str, row: Mapping[str, Any]) -> None:
        keys = list(row.keys())
        placeholders = ",".join(["?"] * len(keys))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(keys)}) VALUES ({placeholders})"
        self.conn.execute(sql, [row[k] for k in keys])

    def write_run(self, row: Mapping[str, Any]) -> None:
        self._ins("runs", row)

    def write_round(self, row: Mapping[str, Any]) -> None:
        self._ins("rounds", row)

    def write_client_round(self, row: Mapping[str, Any]) -> None:
        self._ins("client_rounds", row)

    def finish(self) -> None:
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None
