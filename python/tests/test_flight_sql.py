"""Arrow Flight SQL endpoint via the ADBC Flight SQL driver.

Needs a live server; Flight SQL listens on $JOULEDB_FLIGHT_URL
(default grpc://127.0.0.1:<HTTP port + 4>).
"""
import json
import os
import uuid

import pyarrow as pa
import pytest

flightsql = pytest.importorskip("adbc_driver_flightsql.dbapi")
URL = os.environ.get("JOULEDB_FLIGHT_URL", "grpc://127.0.0.1:8084")


@pytest.fixture()
def conn():
    c = flightsql.connect(URL)
    yield c
    c.close()


def test_ddl_dml_and_query(conn):
    t = "fl_" + uuid.uuid4().hex[:8]
    cur = conn.cursor()
    cur.execute(f"CREATE TABLE {t} (id INT, name TEXT, v FLOAT)")
    cur.execute(f"INSERT INTO {t} VALUES (1, 'a', 1.5), (2, NULL, 2.5), (3, '007', 3.0)")
    cur.execute(f"SELECT id, name, v FROM {t} ORDER BY id")
    tbl = cur.fetch_arrow_table()
    assert tbl.num_rows == 3
    assert tbl.schema.field("id").type == pa.int64()
    assert tbl.schema.field("v").type == pa.float64()
    assert tbl.column("name").to_pylist() == ["a", None, "007"]
    cur.execute(f"SELECT SUM(v) AS s, COUNT(*) AS n FROM {t}")
    assert cur.fetchall() == [(7.0, 3)]
    cur.close()


def test_receipt_in_flight_info():
    flight = pytest.importorskip("pyarrow.flight")
    from adbc_driver_flightsql import dbapi  # noqa: F401  (driver present)
    # Raw Flight SQL client: CommandStatementQuery packed in google.protobuf.Any.
    def varint(n):
        out = b""
        while True:
            b = n & 0x7F
            n >>= 7
            out += bytes([b | (0x80 if n else 0)])
            if not n:
                return out
    sql = b"SELECT 1 AS one"
    cmd = b"\x0a" + varint(len(sql)) + sql
    url = b"type.googleapis.com/arrow.flight.protocol.sql.CommandStatementQuery"
    anyv = b"\x0a" + varint(len(url)) + url + b"\x12" + varint(len(cmd)) + cmd
    client = flight.FlightClient(URL.replace("grpc://", "grpc+tcp://"))
    info = client.get_flight_info(flight.FlightDescriptor.for_command(anyv))
    receipt = json.loads(info.app_metadata)
    assert receipt["energy_joules"] is not None and receipt["energy_joules"] >= 0
    assert receipt["device_target"]
    tbl = client.do_get(info.endpoints[0].ticket).read_all()
    assert tbl.to_pylist() == [{"one": 1}]


def test_error_is_reported(conn):
    cur = conn.cursor()
    with pytest.raises(Exception):
        cur.execute("SELECT * FROM no_such_table_flight")
