"""JouleDB Python SDK.

    import jouledb
    db = jouledb.connect("http://localhost:8080")
    r = db.query("SELECT region, SUM(v) FROM t GROUP BY region")
    r.arrow      # pyarrow.Table
    r.to_pandas()
    r.to_polars()  # zero-copy from Arrow
    r.receipt    # {'energy_joules': ..., 'device_target': 'cpu', ...}

Energy is a receipt, never a refusal: every query runs and reports what
it cost.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pyarrow as pa

from ._native import __version__ as _native_version
from ._native import decode as _decode

__all__ = ["connect", "Connection", "Result", "JouleDBError", "__version__"]
__version__ = "0.1.0"


class JouleDBError(RuntimeError):
    """Server-side or transport error."""


def _column(values: List[Any]) -> pa.Array:
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        # mixed ints and floats -> float64; anything else -> string
        try:
            return pa.array(values, type=pa.float64())
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            return pa.array([None if v is None else str(v) for v in values], type=pa.string())


@dataclass
class Result:
    """One query result: an Arrow table plus the energy receipt."""

    arrow: pa.Table
    receipt: Dict[str, Any] = field(default_factory=dict)
    affected_rows: Optional[int] = None

    def to_pandas(self):
        return self.arrow.to_pandas()

    def to_polars(self):
        import polars as pl

        return pl.from_arrow(self.arrow)

    def to_pylist(self) -> List[Dict[str, Any]]:
        return self.arrow.to_pylist()

    @property
    def num_rows(self) -> int:
        return self.arrow.num_rows

    def __len__(self) -> int:
        return self.arrow.num_rows


class Connection:
    """HTTP connection to a JouleDB server (`POST /api/v1/query`)."""

    def __init__(self, url: str = "http://localhost:8080", token: Optional[str] = None, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def query(self, sql: str) -> Result:
        body = json.dumps({"sql": sql}).encode()
        req = urllib.request.Request(self.url + "/api/v1/query", data=body, method="POST")
        req.add_header("content-type", "application/json")
        if self.token:
            req.add_header("authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("error") or raw.decode(errors="replace")
            except ValueError:
                msg = raw.decode(errors="replace")
            raise JouleDBError(f"HTTP {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            raise JouleDBError(f"cannot reach {self.url}: {e.reason}") from None
        try:
            cols, data, receipt, affected = _decode(raw)
        except RuntimeError as e:
            raise JouleDBError(str(e)) from None
        table = pa.table({c: _column(v) for c, v in zip(cols, data)}) if cols else pa.table({})
        return Result(arrow=table, receipt=receipt, affected_rows=affected)

    execute = query

    def close(self) -> None:  # stateless HTTP; kept for DB-API-style use
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def connect(url: str = "http://localhost:8080", token: Optional[str] = None, timeout: float = 60.0) -> Connection:
    return Connection(url, token=token, timeout=timeout)
