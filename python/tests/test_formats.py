"""Parquet / Arrow IPC round trips with DuckDB and pyarrow.

Needs a live server started with JOULEDB_COPY_DIR, the same directory in
$JOULEDB_COPY_DIR here, and pgwire on $JOULEDB_PG_PORT (HTTP port + 2).
"""
import io
import json
import os
import urllib.request
import uuid

import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq
import pytest

D = os.environ.get("JOULEDB_COPY_DIR")
URL = os.environ.get("JOULEDB_URL", "http://localhost:8080")
PG = os.environ.get("JOULEDB_PG_PORT", "8082")
duckdb = pytest.importorskip("duckdb")
psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.skipif(not D, reason="JOULEDB_COPY_DIR not set")


@pytest.fixture()
def pg():
    c = psycopg.connect(f"host=127.0.0.1 port={PG} user=jouledb dbname=jouledb", autocommit=True)
    yield c
    c.close()


def name():
    return "f_" + uuid.uuid4().hex[:8]


def test_duckdb_parquet_in_and_out(pg):
    t = name()
    src = f"{D}/{t}.parquet"
    duckdb.sql(f"COPY (SELECT i AS id, 'n'||(i%5) AS name, i*0.5 AS v FROM range(20000) t(i)) TO '{src}' (FORMAT parquet)")
    pg.execute(f"CREATE TABLE {t} (id INT, name TEXT, v FLOAT)")
    assert pg.execute(f"COPY {t} FROM '{src}'").statusmessage == "COPY 20000"
    got = pg.execute(f"SELECT COUNT(*), SUM(v), COUNT(DISTINCT name) FROM {t}").fetchone()
    want = duckdb.sql(f"SELECT COUNT(*), SUM(v)::DOUBLE, COUNT(DISTINCT name) FROM '{src}'").fetchone()
    assert tuple(got) == tuple(want)
    out = f"{D}/{t}_out.parquet"
    pg.execute(f"COPY (SELECT id, name, v FROM {t} WHERE id < 1000) TO '{out}'")
    assert duckdb.sql(f"SELECT COUNT(*), SUM(id) FROM '{out}'").fetchone() == (1000, 499500)
    assert pq.read_schema(out).types == [pa.int64(), pa.string(), pa.float64()]


def test_pyarrow_ipc_in_and_out(pg):
    t = name()
    tbl = pa.table({"id": pa.array(range(500), pa.int64()), "s": [None if i % 10 == 0 else f"s{i}" for i in range(500)]})
    pf.write_feather(tbl, f"{D}/{t}.arrow")  # lz4-compressed IPC file
    pg.execute(f"CREATE TABLE {t} (id INT, s TEXT)")
    assert pg.execute(f"COPY {t} FROM '{t}.arrow'").statusmessage == "COPY 500"
    assert pg.execute(f"SELECT COUNT(s) FROM {t}").fetchone()[0] == 450
    pg.execute(f"COPY {t} TO '{D}/{t}_out.arrow'")
    back = pf.read_table(f"{D}/{t}_out.arrow")
    assert back.num_rows == 500 and back.column("s").null_count == 50


def test_http_export(pg):
    t = name()
    pg.execute(f"CREATE TABLE {t} (id INT, v FLOAT)")
    pg.execute(f"INSERT INTO {t} VALUES (1, 1.5), (2, 2.5)")
    b = urllib.request.urlopen(f"{URL}/api/v1/export/{t}?format=parquet").read()
    assert pq.read_table(io.BytesIO(b)).num_rows == 2
    r = urllib.request.Request(
        f"{URL}/api/v1/export",
        data=json.dumps({"sql": f"SELECT SUM(v) AS s FROM {t}", "format": "arrows"}).encode(),
        headers={"content-type": "application/json"},
    )
    assert pa.ipc.open_stream(urllib.request.urlopen(r).read()).read_all().to_pylist() == [{"s": 4.0}]


def test_outside_copy_dir_refused(pg):
    with pytest.raises(psycopg.Error, match="outside JOULEDB_COPY_DIR"):
        pg.execute("COPY x FROM '/etc/hostname.parquet'")


def test_types_kept_from_source(pg):
    import datetime as dt
    import decimal

    t = name()
    tbl = pa.table({
        "zip": pa.array(["02139", "123", None]),
        "d": pa.array([dt.date(2024, 1, 1), None, dt.date(1999, 12, 31)]),
        "ts": pa.array([dt.datetime(2024, 1, 2, 3, 4, 5, 123000), None, None], pa.timestamp("us")),
        "price": pa.array([decimal.Decimal("123.45"), None, decimal.Decimal("-0.01")], pa.decimal128(10, 2)),
        "ok": pa.array([True, None, False]),
        "n": pa.array([None, None, None], pa.null()),
    })
    pq.write_table(tbl, f"{D}/{t}.parquet")
    pg.execute(f"CREATE TABLE {t} (zip TEXT, d TEXT, ts TEXT, price FLOAT, ok BOOLEAN, n TEXT)")
    assert pg.execute(f"COPY {t} FROM '{t}.parquet'").statusmessage == "COPY 3"
    rows = pg.execute(f"SELECT zip, d, ts, price, ok, n FROM {t}").fetchall()
    rows = sorted(rows, key=lambda r: str(r[0]))
    zips = sorted([r[0] for r in rows if r[0] is not None])
    assert zips == ["02139", "123"], zips  # numeric-looking strings stay strings
    by_zip = {r[0]: r for r in rows}
    assert by_zip["02139"][1] == "2024-01-01"
    assert by_zip["02139"][2] == "2024-01-02T03:04:05.123"
    assert abs(float(by_zip["02139"][3]) - 123.45) < 1e-9
    assert by_zip["02139"][4] in (True, "true", "t")
    assert all(r[5] is None for r in rows)
    # zip is still text on the way back out
    pg.execute(f"COPY {t} TO '{D}/{t}_back.parquet'")
    assert pq.read_table(f"{D}/{t}_back.parquet").column("zip").type == pa.string()


def test_copy_file_in_transaction(pg):
    t = name()
    pq.write_table(pa.table({"id": pa.array(range(300), pa.int64()), "s": [f"0{i}" for i in range(300)]}), f"{D}/{t}.parquet")
    pg.execute(f"CREATE TABLE {t} (id INT, s TEXT)")
    pg.autocommit = False
    try:
        # (plain INSERTs inside a transaction aren't visible to the session's own
        # SELECT until COMMIT yet; COPY behaves the same, so no in-txn count here)
        assert pg.execute(f"COPY {t} FROM '{t}.parquet'").statusmessage == "COPY 300"
        pg.rollback()
        assert pg.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0
        pg.rollback()
        pg.execute(f"COPY {t} FROM '{t}.parquet'")
        pg.commit()
    finally:
        pg.rollback()
        pg.autocommit = True
    assert pg.execute(f"SELECT COUNT(*), MIN(s) FROM {t}").fetchone() == (300, "00")
