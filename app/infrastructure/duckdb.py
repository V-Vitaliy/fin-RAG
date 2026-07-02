from __future__ import annotations

import asyncio
from pathlib import Path

import duckdb
from duckdb import DuckDBPyConnection


class DuckDBManager:
    """
    Small infrastructure wrapper around a shared DuckDB file.

    Important:
    - Do not keep one global DuckDB connection open forever.
    - Use connect_readonly() per query/tool call.
    - Use connect_writer() only inside controlled ingestion/write paths.
    - write_lock protects the shared DuckDB file from concurrent writers.
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.write_lock = asyncio.Lock()

    def ensure_parent_dir(self) -> None:
        parent = Path(self.path).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)

    def connect_readonly(self) -> DuckDBPyConnection:
        if not Path(self.path).exists():
            raise FileNotFoundError(f"DuckDB file does not exist: {self.path}")

        return duckdb.connect(self.path, read_only=True)

    def connect_writer(self) -> DuckDBPyConnection:
        self.ensure_parent_dir()
        return duckdb.connect(self.path)